"""
Improved V-net for fringe pattern denoising (handles speckle + 2nd harmonic)
Data: fringe_data_v2.mat (500 samples, 128x128, complex noise)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# 固定随机种子（可选）
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# -------------------- 1. 加载数据 --------------------
mat_path = r'C:/Users/25714/Documents/MATLAB/fringe_data_v2.mat'
data = sio.loadmat(mat_path)
X = torch.tensor(data['XTrain'], dtype=torch.float32)  # [H,W,C,N]
Y = torch.tensor(data['YTrain'], dtype=torch.float32)

# 转置为 [N, C, H, W]
X = X.permute(3, 2, 0, 1)
Y = Y.permute(3, 2, 0, 1)

print(f"数据形状: X={X.shape}, Y={Y.shape}")

# -------------------- 2. 数据增强 --------------------
class RandomAugment:
    def __init__(self, p_flip=0.5, p_rot=0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot

    def __call__(self, x, y):
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        if torch.rand(1) < self.p_rot:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1,2])
            y = torch.rot90(y, k, dims=[1,2])
        return x, y

augment = RandomAugment()

# -------------------- 3. 改进版 V-net（宽通道，适合大图） --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.double_conv(x)
        pooled = self.pool(feat)
        return pooled, feat

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, dropout)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class ImprovedVNet(nn.Module):
    """
    改进版 V-net:
    - 编码器通道数递增 (64->128->256->512)，提高特征提取能力
    - 解码器通道数递减 (512->256->128->64)
    - 加入 Dropout (0.1) 防止过拟合
    - 适合处理 128x128 带复杂噪声的干涉图
    """
    def __init__(self, in_ch=1, out_ch=1, dropout=0.1):
        super().__init__()
        # 编码器 (通道递增)
        self.inc = DoubleConv(in_ch, 64, dropout)
        self.down1 = DownBlock(64, 128, dropout)
        self.down2 = DownBlock(128, 256, dropout)
        self.down3 = DownBlock(256, 512, dropout)
        self.down4 = DownBlock(512, 512, dropout)   # bottleneck

        # 解码器 (通道递减)
        self.up1 = UpBlock(512 + 512, 256, dropout)
        self.up2 = UpBlock(256 + 256, 128, dropout)
        self.up3 = UpBlock(128 + 128, 64, dropout)
        self.up4 = UpBlock(64 + 64, 64, dropout)

        self.outc = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # encoder
        x0 = self.inc(x)           # 64
        x1, p1 = self.down1(x0)    # 128
        x2, p2 = self.down2(x1)    # 256
        x3, p3 = self.down3(x2)    # 512
        x4, p4 = self.down4(x3)    # 512

        # decoder
        d1 = self.up1(x4, x3)      # 512+512 -> 256
        d2 = self.up2(d1, x2)      # 256+256 -> 128
        d3 = self.up3(d2, x1)      # 128+128 -> 64
        d4 = self.up4(d3, x0)      # 64+64 -> 64

        return self.outc(d4)

# -------------------- 4. 训练准备 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedVNet(in_ch=1, out_ch=1, dropout=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
criterion = nn.L1Loss()   # L1 loss 保留边缘

# 划分训练/验证集 (90%/10%)
n_samples = X.shape[0]
n_train = int(0.9 * n_samples)
indices = np.random.permutation(n_samples)
train_idx, val_idx = indices[:n_train], indices[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

batch_size = 16   # 如果显存不够，可减到8
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

epochs = 200
best_val_loss = float('inf')
train_losses = []
val_losses = []

print(f"训练设备: {device}")
print(f"训练样本: {len(X_train)}, 验证样本: {len(X_val)}")

# -------------------- 5. 训练循环 --------------------
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for batch_x, batch_y in train_loader:
        # 数据增强
        batch_x_aug, batch_y_aug = batch_x.clone(), batch_y.clone()
        for i in range(batch_x.size(0)):
            batch_x_aug[i], batch_y_aug[i] = augment(batch_x_aug[i], batch_y_aug[i])

        batch_x_aug = batch_x_aug.to(device)
        batch_y_aug = batch_y_aug.to(device)

        optimizer.zero_grad()
        pred = model(batch_x_aug)
        loss = criterion(pred, batch_y_aug)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_vnet_improved.pth')
        print(f"Epoch {epoch+1}: 保存最佳模型 (val_loss={avg_val_loss:.6f})")

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# -------------------- 6. 测试与可视化 --------------------
# 加载最佳模型
model.load_state_dict(torch.load('best_vnet_improved.pth'))
model.eval()

# 取第一个验证样本测试
test_input = X_val[0:1].to(device)
test_truth = Y_val[0].cpu().squeeze().numpy()
with torch.no_grad():
    test_output = model(test_input)
test_output_np = test_output.cpu().squeeze().numpy()
test_input_np = X_val[0].cpu().squeeze().numpy()

# 保存结果
sio.savemat('result_vnet_improved.mat', {
    'noisy': test_input_np,
    'denoised': test_output_np,
    'clean': test_truth,
    'train_loss': train_losses,
    'val_loss': val_losses
})

def psnr(img1, img2, peak=2.0):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(peak / np.sqrt(mse))

psnr_val = psnr(test_output_np, test_truth)
print(f"测试图像 PSNR: {psnr_val:.2f} dB")

# 绘图
plt.figure(figsize=(15, 10))
plt.subplot(2,3,1)
plt.imshow(test_input_np, cmap='gray')
plt.title('Noisy Input')
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(test_output_np, cmap='gray')
plt.title(f'Improved V-net (PSNR={psnr_val:.2f}dB)')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(test_truth, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(2,3,4)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training Curves (Improved V-net)')
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.yscale('log')

plt.subplot(2,3,5)
error_map = np.abs(test_output_np - test_truth)
plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.title('Absolute Error Map')

plt.tight_layout()
plt.savefig('result_vnet_improved.png', dpi=150)
plt.show()

print("改进版 V-net 训练完成！结果保存为 result_vnet_improved.mat 和 result_vnet_improved.png")