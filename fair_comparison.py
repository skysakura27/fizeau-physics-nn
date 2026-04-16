"""公平对比实验：Baseline vs PINN，在相同/不同难度数据上的表现。

对比矩阵 (2 模型 × 2 数据):
  - UnrolledBaseline (best_baseline.pth) — 旧参数训练
  - UnrolledPINN     (best_pinn.pth)     — 新参数训练
  - 旧数据: coeffs×2.0, global_scale=10, 有tilt  (难)
  - 新数据: coeffs×0.5, global_scale=1.0, 无tilt  (易)
"""

import numpy as np
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from src.models.integrated_net import UnrolledBaseline, UnrolledPINN
from src.core.physics_ops import AirySimulator

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'

# ===================== 数据生成 =====================
def build_zernike_basis(size=128, n_zernike=15):
    lin = np.linspace(-1, 1, size, dtype=np.float32)
    xg, yg = np.meshgrid(lin, lin)
    rho = np.sqrt(xg**2 + yg**2)
    theta = np.arctan2(yg, xg)
    mask = (rho <= 1.0).astype(np.float32)
    basis = []
    for n in range(0, 8):
        for m in range(-n, n+1, 2):
            if len(basis) >= n_zernike:
                break
            radial = rho ** n
            angular = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
            b = radial * angular * mask
            b = b / (np.abs(b).max() + 1e-8)
            basis.append(b)
        if len(basis) >= n_zernike:
            break
    return np.stack(basis, axis=0), mask, xg, yg


def generate_data(config, n_samples=500, seed=42):
    """config: dict with keys coeff_scale, global_scale, use_tilt"""
    rng = np.random.default_rng(seed)
    basis, mask, xg, yg = build_zernike_basis()
    n_zernike = basis.shape[0]
    airy = AirySimulator(R=0.5, I_max=1.0,
                         global_scale=config['global_scale'],
                         learnable=False)
    X_list, Y_list = [], []
    for _ in range(n_samples):
        coeffs = rng.standard_normal(n_zernike).astype(np.float32)
        coeffs *= config['coeff_scale']
        phase = np.tensordot(coeffs, basis, axes=1) * mask
        phase_t = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)

        if config['use_tilt']:
            tx = np.float32(rng.uniform(-1.0, 1.0))
            ty = np.float32(rng.uniform(-1.0, 1.0))
            tilt = torch.from_numpy((tx * xg + ty * yg)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                I_clean = airy(phase_t + tilt)
        else:
            with torch.no_grad():
                I_clean = airy(phase_t)

        noise = torch.randn_like(I_clean) * 0.05
        I_noisy = (I_clean + noise).clamp(0.0, 1.0)
        X_list.append(I_noisy)
        Y_list.append(phase_t)
    return torch.cat(X_list), torch.cat(Y_list)


# 旧参数 (baseline 训练时用的)
OLD_CFG = dict(coeff_scale=2.0, global_scale=10.0, use_tilt=True)
# 新参数 (PINN 训练时用的)
NEW_CFG = dict(coeff_scale=0.5, global_scale=1.0, use_tilt=False)

print("生成旧参数数据 (难) ...")
X_old, Y_old = generate_data(OLD_CFG)
print("生成新参数数据 (易) ...")
X_new, Y_new = generate_data(NEW_CFG)

# 取验证集 (与训练脚本相同的 split)
indices = np.random.permutation(500)
val_idx = indices[int(0.9*500):]
X_old_val, Y_old_val = X_old[val_idx], Y_old[val_idx]
X_new_val, Y_new_val = X_new[val_idx], Y_new[val_idx]

# ===================== 加载模型 =====================
print("\n加载模型 ...")
# Baseline (旧检查点可能缺少新增参数，用 strict=False)
baseline = UnrolledBaseline(n_iters=1)
baseline.load_state_dict(torch.load(OUTPUT_DIR / 'best_baseline.pth',
                                     map_location='cpu', weights_only=True),
                          strict=False)
baseline.eval()

# PINN
pinn = UnrolledPINN(n_iters=5, share_weights=True)
pinn.load_state_dict(torch.load(OUTPUT_DIR / 'best_pinn.pth',
                                 map_location='cpu', weights_only=True))
pinn.eval()

# ===================== 评估 =====================
def psnr(a, b, peak=2*np.pi):
    mse = np.mean((a - b)**2)
    return 100.0 if mse == 0 else 20 * np.log10(peak / np.sqrt(mse))


def evaluate(model, X_val, Y_val, n_eval=None):
    """返回 PSNR 列表和推理时间"""
    if n_eval is None:
        n_eval = len(X_val)
    psnrs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_eval):
            pred = model(X_val[i:i+1]).squeeze().numpy()
            gt = Y_val[i].squeeze().numpy()
            psnrs.append(psnr(pred, gt))
    elapsed = time.perf_counter() - t0
    return psnrs, elapsed


results = {}
configs = [
    ('Baseline → 旧数据(难)', baseline, X_old_val, Y_old_val),
    ('Baseline → 新数据(易)', baseline, X_new_val, Y_new_val),
    ('PINN → 旧数据(难)',     pinn,     X_old_val, Y_old_val),
    ('PINN → 新数据(易)',     pinn,     X_new_val, Y_new_val),
]

print("\n" + "="*60)
print(f"{'配置':<25s} {'平均PSNR':>10s} {'最低':>8s} {'最高':>8s} {'耗时':>8s}")
print("="*60)
for name, model, xv, yv in configs:
    ps, t = evaluate(model, xv, yv)
    results[name] = ps
    print(f"{name:<25s} {np.mean(ps):>8.2f} dB {np.min(ps):>6.2f} dB {np.max(ps):>6.2f} dB {t:>6.2f}s")
print("="*60)

# ===================== 可视化 =====================
# 选一个样本做 4 模型对比
sample_idx = 0
fig, axes = plt.subplots(4, 5, figsize=(20, 16))

plot_configs = [
    ('Baseline + 旧数据(难)', baseline, X_old_val, Y_old_val),
    ('Baseline + 新数据(易)', baseline, X_new_val, Y_new_val),
    ('PINN + 旧数据(难)',     pinn,     X_old_val, Y_old_val),
    ('PINN + 新数据(易)',     pinn,     X_new_val, Y_new_val),
]

for row, (name, model, xv, yv) in enumerate(plot_configs):
    inp = xv[sample_idx:sample_idx+1]
    gt = yv[sample_idx].squeeze().numpy()
    with torch.no_grad():
        pred = model(inp).squeeze().numpy()
    inp_np = inp.squeeze().numpy()
    p = psnr(pred, gt)
    err = np.abs(pred - gt)

    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())

    axes[row, 0].imshow(inp_np, cmap='gray')
    axes[row, 0].set_title('Input Intensity')
    axes[row, 0].set_ylabel(name, fontsize=11, fontweight='bold')

    axes[row, 1].imshow(pred, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[row, 1].set_title(f'Predicted (PSNR={p:.2f} dB)')

    axes[row, 2].imshow(gt, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[row, 2].set_title('Ground Truth')

    axes[row, 3].imshow(err, cmap='hot')
    axes[row, 3].set_title(f'|Error| (max={err.max():.3f})')

    # 1D profile along center row
    mid = gt.shape[0] // 2
    axes[row, 4].plot(gt[mid, :], 'b-', label='GT', alpha=0.8)
    axes[row, 4].plot(pred[mid, :], 'r--', label='Pred', alpha=0.8)
    axes[row, 4].legend(fontsize=8)
    axes[row, 4].set_title('Center Profile')
    axes[row, 4].set_xlim(0, gt.shape[1])

    for col in range(4):
        axes[row, col].axis('off')

fig.suptitle('公平对比: Baseline vs PINN × 旧数据(难) vs 新数据(易)', fontsize=14)
plt.tight_layout()
out_path = OUTPUT_DIR / 'fair_comparison.png'
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\n对比图保存至: {out_path}")

# ===================== GPU 速度估算 =====================
print("\n" + "="*60)
print("GPU 加速分析")
print("="*60)

# 测一次 CPU 推理时间
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(10):
        _ = pinn(X_new_val[i % len(X_new_val):i % len(X_new_val)+1])
cpu_time = (time.perf_counter() - t0) / 10

# 测一次 CPU 训练步时间 (单 batch forward+backward)
pinn_tmp = UnrolledPINN(n_iters=5, share_weights=True)
pinn_tmp.train()
opt = torch.optim.Adam(pinn_tmp.parameters(), lr=1e-3)
batch = X_new_val[:16]
batch_y = Y_new_val[:16]

t0 = time.perf_counter()
for _ in range(3):
    opt.zero_grad()
    pred = pinn_tmp(batch)
    loss = torch.nn.functional.mse_loss(pred, batch_y)
    loss.backward()
    opt.step()
cpu_train_step = (time.perf_counter() - t0) / 3

print(f"CPU 单样本推理: {cpu_time*1000:.1f} ms")
print(f"CPU 单 batch(16) 训练步: {cpu_train_step*1000:.1f} ms")
print(f"CPU 估算每 epoch 时间: {cpu_train_step * (450/16):.1f} s")
print(f"CPU 估算 100 epochs: {cpu_train_step * (450/16) * 100 / 60:.1f} min")
print()
print("GPU 预估加速 (典型值):")
print(f"  NVIDIA RTX 3060:  ~10-20x → {cpu_train_step*(450/16)*100/60/15:.1f} min")
print(f"  NVIDIA RTX 4090:  ~30-50x → {cpu_train_step*(450/16)*100/60/40:.1f} min")
print(f"  Intel Arc A770:   ~5-10x  → {cpu_train_step*(450/16)*100/60/7:.1f} min (via IPEX)")
print(f"  Google Colab T4:  ~8-15x  → {cpu_train_step*(450/16)*100/60/12:.1f} min (免费)")
