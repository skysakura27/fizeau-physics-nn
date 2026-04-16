"""从当前 best_pinn.pth 检查点生成结果可视化（不需要等训练完成）。"""

import numpy as np
from pathlib import Path
import torch
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.models.integrated_net import UnrolledPINN
from src.core.physics_ops import AirySimulator

OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'

# -------- 1. 用相同种子生成验证数据 --------
def generate_synthetic_dataset(
    n_samples=500, size=128, R=0.5, I_max=1.0,
    noise_std=0.05, n_zernike=15, seed=42,
):
    rng = np.random.default_rng(seed)
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
    basis = np.stack(basis, axis=0)

    airy = AirySimulator(R=R, I_max=I_max, global_scale=1.0, learnable=False)
    X_list, Y_list = [], []
    for _ in range(n_samples):
        coeffs = rng.standard_normal(n_zernike).astype(np.float32)
        coeffs *= 0.5
        phase = np.tensordot(coeffs, basis, axes=1) * mask
        phase_t = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            I_clean = airy(phase_t)
        noise = torch.randn_like(I_clean) * noise_std
        I_noisy = (I_clean + noise).clamp(0.0, I_max)
        X_list.append(I_noisy)
        Y_list.append(phase_t)
    return torch.cat(X_list), torch.cat(Y_list)

print("生成合成数据 ...")
X, Y = generate_synthetic_dataset()

# 用相同随机分割
n_train = int(0.9 * 500)
indices = np.random.permutation(500)
val_idx = indices[n_train:]
X_val, Y_val = X[val_idx], Y[val_idx]

# -------- 2. 加载模型 --------
ckpt_path = OUTPUT_DIR / 'best_pinn.pth'
print(f"加载检查点: {ckpt_path}")
model = UnrolledPINN(n_iters=5, share_weights=True)
state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
model.load_state_dict(state)
model.eval()

# 打印学习到的物理参数
airy = model.airy
R_val = airy.R.item()
I_max_val = airy.I_max.item()
gs_val = airy.global_scale.item()
print(f"学习参数: R={R_val:.4f}, I_max={I_max_val:.4f}, global_scale={gs_val:.4f}")

# -------- 3. 多样本推理 --------
n_show = min(4, len(X_val))
def psnr(a, b, peak=2*np.pi):
    mse = np.mean((a - b)**2)
    return 100.0 if mse == 0 else 20 * np.log10(peak / np.sqrt(mse))

fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show))
if n_show == 1:
    axes = axes[np.newaxis, :]

psnr_list = []
for i in range(n_show):
    inp = X_val[i:i+1]
    gt  = Y_val[i].squeeze().numpy()
    with torch.no_grad():
        pred = model(inp).squeeze().numpy()
    inp_np = inp.squeeze().numpy()
    p = psnr(pred, gt)
    psnr_list.append(p)

    vmin = min(pred.min(), gt.min())
    vmax = max(pred.max(), gt.max())

    axes[i, 0].imshow(inp_np, cmap='gray')
    axes[i, 0].set_title(f'Input Intensity #{i}')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(pred, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[i, 1].set_title(f'Predicted (PSNR={p:.2f} dB)')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(gt, cmap='RdBu', vmin=vmin, vmax=vmax)
    axes[i, 2].set_title('Ground Truth')
    axes[i, 2].axis('off')

    err = np.abs(pred - gt)
    axes[i, 3].imshow(err, cmap='hot')
    axes[i, 3].set_title(f'Error (max={err.max():.3f})')
    axes[i, 3].axis('off')

fig.suptitle(f'UnrolledPINN — Mean PSNR={np.mean(psnr_list):.2f} dB  (checkpoint: best_pinn.pth)', fontsize=14)
plt.tight_layout()
out_path = OUTPUT_DIR / 'result_pinn.png'
plt.savefig(out_path, dpi=150)
plt.close()

# -------- 4. 保存 .mat --------
# 用第一个样本
inp0 = X_val[0:1]
gt0  = Y_val[0].squeeze().numpy()
with torch.no_grad():
    pred0 = model(inp0).squeeze().numpy()
sio.savemat(str(OUTPUT_DIR / 'result_pinn.mat'), {
    'intensity_input': inp0.squeeze().numpy(),
    'pred_phase': pred0,
    'gt_phase': gt0,
})

print(f"\n===== 结果 =====")
for i, p in enumerate(psnr_list):
    print(f"  样本 {i}: PSNR = {p:.2f} dB")
print(f"  平均 PSNR = {np.mean(psnr_list):.2f} dB")
print(f"结果保存至: {out_path}")
print(f"MAT 保存至: {OUTPUT_DIR / 'result_pinn.mat'}")
