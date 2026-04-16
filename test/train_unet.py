import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------- 1. 加载数据 --------------------
mat_path = r'C:/Users/25714/Documents/MATLAB/fringe_data.mat'   # 你的MATLAB数据路径
data = sio.loadmat(mat_path)
X = torch.tensor(data['XTrain'], dtype=torch.float32)  # [H,W,C,N]
Y = torch.tensor(data['YTrain'], dtype=torch.float32)

# 转置为 [N, C, H, W]
X = X.permute(3, 2, 0, 1)
Y = Y.permute(3, 2, 0, 1)

print(f"数据形状: X={X.shape}, Y={Y.shape}")

# -------------------- 2. 数据增强函数 --------------------
class RandomAugment:
    def __init__(self, p_flip=0.5, p_rot=0.5):
        self.p_flip = p_flip
        self.p_rot = p_rot
    
    def __call__(self, x, y):
        # 随机水平翻转
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[2])
        # 随机垂直翻转
        if torch.rand(1) < self.p_flip:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[1])
        # 随机旋转 90/180/270 度
        if torch.rand(1) < self.p_rot:
            k = torch.randint(1, 4, (1,)).item()
            x = torch.rot90(x, k, dims=[1,2])
            y = torch.rot90(y, k, dims=[1,2])
        return x, y

augment = RandomAugment()

# -------------------- 3. 定义 U-Net --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if not mid_ch:
            mid_ch = out_ch
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# -------------------- 4. 训练设置 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(n_channels=1, n_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
criterion = nn.L1Loss()

# 数据集划分（训练集90%，验证集10%）
n_samples = X.shape[0]
n_train = int(0.9 * n_samples)
indices = np.random.permutation(n_samples)
train_idx, val_idx = indices[:n_train], indices[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

batch_size = 16
train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

epochs = 100   # 可改成150或200
best_val_loss = float('inf')

train_losses = []
val_losses = []

print(f"训练设备: {device}")
print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

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
    
    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_unet.pth')
        print(f"Epoch {epoch+1}: 保存最佳模型 (val_loss={avg_val_loss:.6f})")
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

# -------------------- 6. 测试并保存结果 --------------------
# 加载最佳模型
model.load_state_dict(torch.load('best_unet.pth'))
model.eval()

# 用第一个验证样本测试
test_input = X_val[0:1].to(device)        # shape: [1,1,128,128]
test_truth = Y_val[0].cpu().squeeze().numpy()   # 变为 [128,128]
with torch.no_grad():
    test_output = model(test_input)       # [1,1,128,128]
test_output_np = test_output.cpu().squeeze().numpy()   # [128,128]
test_input_np = X_val[0].cpu().squeeze().numpy()       # [128,128]

# 保存MATLAB文件
sio.savemat('result_optimized.mat', {
    'noisy': test_input_np,
    'denoised': test_output_np,
    'clean': test_truth,
    'train_loss': train_losses,
    'val_loss': val_losses
})

# 计算 PSNR
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
plt.title(f'Denoised Output (PSNR={psnr_val:.2f}dB)')
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(test_truth, cmap='gray')
plt.title('Ground Truth')
plt.axis('off')

plt.subplot(2,3,4)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Training Curves')
plt.xlabel('Epoch')
plt.ylabel('L1 Loss')
plt.yscale('log')

plt.subplot(2,3,5)
error_map = np.abs(test_output_np - test_truth)
plt.imshow(error_map, cmap='hot')
plt.colorbar()
plt.title('Absolute Error Map')

plt.tight_layout()
plt.savefig('result_optimized.png', dpi=150)
plt.show()

print("优化训练完成！结果已保存为 result_optimized.mat 和 result_optimized.png")