"""
Training script for UnrolledBaseline — Fizeau phase retrieval PINN.
Based on: test/train_vnet.py (Reyes-Figueroa et al., Applied Optics, 2021)

Model  : UnrolledBaseline (src.models.integrated_net)
Input  : X — measured intensity fringe pattern  (B, 1, 128, 128)
Output : predicted phase map                     (B, 1, 128, 128)
Target : Y — ground truth phase map              (B, 1, 128, 128)
"""

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.models.integrated_net import UnrolledBaseline
from src.core.physics_ops import AirySimulator

# -------------------- 输出目录 --------------------
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- 1. 合成训练数据 --------------------
def generate_synthetic_dataset(
    n_samples: int = 500,
    size: int = 128,
    R: float = 0.5,
    I_max: float = 1.0,
    noise_std: float = 0.05,
    n_zernike: int = 15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate physics-informed synthetic intensity/phase pairs.

    For each sample:
      1. Build a random Zernike-like phase map (ground truth Y).
      2. Apply the Airy formula to get a clean intensity image.
      3. Add random tilt + Gaussian noise to get noisy intensity (input X).

    Returns:
        X: (N, 1, H, W) noisy intensity images.
        Y: (N, 1, H, W) ground truth phase maps.
    """
    rng = np.random.default_rng(seed)

    # Normalised pupil coordinates on a unit disk
    lin = np.linspace(-1, 1, size, dtype=np.float32)
    xg, yg = np.meshgrid(lin, lin)
    rho = np.sqrt(xg ** 2 + yg ** 2)
    theta = np.arctan2(yg, xg)
    mask = (rho <= 1.0).astype(np.float32)          # circular pupil

    # Pre-compute a set of Zernike-like radial basis functions:
    #   Z_j(rho, theta) = rho^n * cos/sin(m * theta)
    # using sequential (n, m) pairs — simple but physically meaningful.
    basis = []  # list of (H, W) arrays
    for n in range(0, 8):
        for m in range(-n, n + 1, 2):
            if len(basis) >= n_zernike:
                break
            radial = rho ** n
            angular = np.cos(m * theta) if m >= 0 else np.sin(-m * theta)
            b = radial * angular * mask
            b = b / (np.abs(b).max() + 1e-8)       # normalise to [-1, 1]
            basis.append(b)
        if len(basis) >= n_zernike:
            break
    basis = np.stack(basis, axis=0)                  # (n_zernike, H, W)

    # Fixed (non-learnable) Airy simulator for data generation
    airy = AirySimulator(R=R, I_max=I_max, learnable=False)

    X_list, Y_list = [], []
    for _ in range(n_samples):
        # Random Zernike coefficients — moderate amplitude
        coeffs = rng.standard_normal(n_zernike).astype(np.float32)
        coeffs *= 2.0                                # scale for ~[-2π, 2π] range
        phase = np.tensordot(coeffs, basis, axes=1)  # (H, W)
        phase *= mask                                # zero outside pupil

        # Random tilt perturbation on the measured intensity
        tilt_x = np.float32(rng.uniform(-1.0, 1.0))
        tilt_y = np.float32(rng.uniform(-1.0, 1.0))
        tilt = tilt_x * xg + tilt_y * yg            # (H, W)

        phase_t = torch.from_numpy(phase).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        tilt_t  = torch.from_numpy(tilt).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            I_clean = airy(phase_t + tilt_t)          # (1,1,H,W)

        # Additive Gaussian noise on intensity
        noise = torch.randn_like(I_clean) * noise_std
        I_noisy = (I_clean + noise).clamp(0.0, I_max)

        X_list.append(I_noisy)
        Y_list.append(phase_t)

    X = torch.cat(X_list, dim=0)   # (N, 1, H, W)
    Y = torch.cat(Y_list, dim=0)   # (N, 1, H, W)
    return X, Y


X, Y = generate_synthetic_dataset(n_samples=500)
print(f"合成数据形状: X={X.shape}, Y={Y.shape}")

# -------------------- 2. 数据增强函数 --------------------
class RandomAugment:
    def __init__(self, p_flip=0.5, p_rot=0.5):
        self.p_flip = p_flip
        self.p_rot  = p_rot

    def __call__(self, x, y):
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        if torch.rand(1) < self.p_rot:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1, 2])
            y = torch.rot90(y, k, dims=[1, 2])
        return x, y

augment = RandomAugment()

# -------------------- 3. 训练准备 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# UnrolledBaseline: inputs intensity X, outputs predicted phase
model     = UnrolledBaseline(n_iters=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.L1Loss()   # L1 loss — preserves phase edges

# 划分训练集 / 验证集 (90% / 10%)
n_samples = X.shape[0]
n_train   = int(0.9 * n_samples)
indices   = np.random.permutation(n_samples)
train_idx, val_idx = indices[:n_train], indices[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]

batch_size    = 16
train_dataset = TensorDataset(X_train, Y_train)
val_dataset   = TensorDataset(X_val,   Y_val)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

epochs        = 150
best_val_loss = float('inf')
train_losses  = []
val_losses    = []

print(f"训练设备: {device}")
print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

# -------------------- 4. 训练循环 --------------------
epoch_bar = tqdm(range(epochs), desc='训练进度', unit='epoch')
for epoch in epoch_bar:
    model.train()
    total_train_loss = 0

    batch_bar = tqdm(
        train_loader, desc=f'Epoch {epoch+1}/{epochs}',
        leave=False, unit='batch',
    )
    for batch_x, batch_y in batch_bar:
        # 在线数据增强
        batch_x_aug = batch_x.clone()
        batch_y_aug = batch_y.clone()
        for i in range(batch_x.size(0)):
            batch_x_aug[i], batch_y_aug[i] = augment(batch_x_aug[i], batch_y_aug[i])

        batch_x_aug = batch_x_aug.to(device)  # intensity input
        batch_y_aug = batch_y_aug.to(device)  # ground truth phase

        optimizer.zero_grad()
        pred_phase = model(batch_x_aug)                    # (B, 1, H, W) predicted phase
        loss       = criterion(pred_phase, batch_y_aug)    # phase error
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        batch_bar.set_postfix(loss=f'{loss.item():.4f}')

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred_phase = model(batch_x)
            loss       = criterion(pred_phase, batch_y)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    lr_now = optimizer.param_groups[0]['lr']

    # 每个 epoch 都更新外层进度条信息
    epoch_bar.set_postfix(
        train=f'{avg_train_loss:.4f}',
        val=f'{avg_val_loss:.4f}',
        lr=f'{lr_now:.1e}',
        best=f'{best_val_loss:.4f}',
    )

    scheduler.step(avg_val_loss)

    saved = ''
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / 'best_baseline.pth')
        saved = ' ★ saved'

    tqdm.write(
        f'[Epoch {epoch+1:>3d}/{epochs}] '
        f'train={avg_train_loss:.5f}  val={avg_val_loss:.5f}  '
        f'lr={lr_now:.1e}{saved}'
    )

# -------------------- 5. 测试与可视化 --------------------
model.load_state_dict(torch.load(OUTPUT_DIR / 'best_baseline.pth'))
model.eval()

test_input  = X_val[0:1].to(device)
test_truth  = Y_val[0].cpu().squeeze().numpy()
with torch.no_grad():
    test_output = model(test_input)
test_output_np = test_output.cpu().squeeze().numpy()
test_input_np  = X_val[0].cpu().squeeze().numpy()

# 保存结果
sio.savemat(str(OUTPUT_DIR / 'result_baseline.mat'), {
    'intensity_input': test_input_np,
    'pred_phase':      test_output_np,
    'gt_phase':        test_truth,
    'train_loss':      train_losses,
    'val_loss':        val_losses,
})

# 计算 PSNR（相位范围约 2π，peak 设为 2π）
def psnr(img1, img2, peak=2 * np.pi):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(peak / np.sqrt(mse))

psnr_val = psnr(test_output_np, test_truth)
print(f"测试图像 PSNR: {psnr_val:.2f} dB")

# 绘图
vmin = min(test_output_np.min(), test_truth.min())
vmax = max(test_output_np.max(), test_truth.max())
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(test_input_np, cmap='gray')
plt.title('Intensity Input (X)')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(test_output_np, cmap='RdBu', vmin=vmin, vmax=vmax)
plt.title(f'Predicted Phase (PSNR={psnr_val:.2f} dB)')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(test_truth, cmap='RdBu', vmin=vmin, vmax=vmax)
plt.title('Ground Truth Phase (Y)')
plt.colorbar()
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.legend()
plt.title('Training Curves (UnrolledBaseline)')
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.yscale('log')

plt.subplot(2, 3, 5)
error_map = np.abs(test_output_np - test_truth)
plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.title('Absolute Phase Error')
plt.axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'result_baseline.png', dpi=150)
plt.show()

print(f"训练完成！结果保存至 {OUTPUT_DIR}")
