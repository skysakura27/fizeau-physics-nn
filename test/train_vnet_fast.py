# train_vnet_fast.py
"""
简化版 V-net 去噪训练 + 单图像推理
特点：
1. 数据泄露防止（训练/验证集严格划分）
2. 可快速训练，小轮数，小 batch
3. 参数易调节：epochs、batch_size
4. 随机验证样本演示
5. 支持单图像推理
6. 中文可视化
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
import argparse
from skimage.transform import resize
from imageio import imread, imwrite
import os

# -------------------- 固定随机种子 --------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)

# -------------------- 参数解析 --------------------
parser = argparse.ArgumentParser()
parser.add_argument('--infer', type=str, default=None, help='单张图像路径 (.png/.jpg/.tif) 用于推理')
parser.add_argument('--model', type=str, default='best_vnet_fast.pth', help='训练好的模型路径')
parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=2, help='训练批大小')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# -------------------- 数据增强 --------------------
class RandomAugment:
    """水平/垂直翻转 + 90/180/270旋转"""
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
            k = torch.randint(1,4,(1,)).item()
            x = torch.rot90(x,k,dims=[1,2])
            y = torch.rot90(y,k,dims=[1,2])
        return x,y
augment = RandomAugment()

# -------------------- 网络结构 (V-net 基础) --------------------
class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1,inplace=True)
        )
    def forward(self,x):
        return self.conv(x)

class DownBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.double_conv = DoubleConv(in_ch,out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self,x):
        feat = self.double_conv(x)
        pooled = self.pool(feat)
        return pooled, feat

class UpBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = DoubleConv(in_ch,out_ch)
    def forward(self,x,skip):
        x = self.up(x)
        # 调整尺寸
        diffY = skip.size()[2]-x.size()[2]
        diffX = skip.size()[3]-x.size()[3]
        x = nn.functional.pad(x,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])
        x = torch.cat([skip,x],dim=1)
        return self.conv(x)

class VNet(nn.Module):
    def __init__(self,in_ch=1,out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch,64)
        self.down1 = DownBlock(64,128)
        self.down2 = DownBlock(128,256)
        self.down3 = DownBlock(256,512)
        self.down4 = DownBlock(512,512)
        self.up1 = UpBlock(512+512,256)
        self.up2 = UpBlock(256+256,128)
        self.up3 = UpBlock(128+128,64)
        self.up4 = UpBlock(64+64,64)
        self.outc = nn.Conv2d(64,out_ch,1)
    def forward(self,x):
        x0 = self.inc(x)
        x1,_ = self.down1(x0)
        x2,_ = self.down2(x1)
        x3,_ = self.down3(x2)
        x4,_ = self.down4(x3)
        d1 = self.up1(x4,x3)
        d2 = self.up2(d1,x2)
        d3 = self.up3(d2,x1)
        d4 = self.up4(d3,x0)
        return self.outc(d4)

# -------------------- 单图像去噪函数 --------------------
def denoise_single_image(model,img_np):
    model.eval()
    h,w = img_np.shape
    target_h = max(512,h)
    target_w = max(512,w)
    img_resized = resize(img_np,(target_h,target_w),mode='reflect',anti_aliasing=True)
    x = torch.tensor(img_resized[np.newaxis,np.newaxis,:,:],dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(x)
    out_np = out.cpu().squeeze().numpy()
    out_resized = resize(out_np,(h,w),mode='reflect',anti_aliasing=True)
    return out_resized

# -------------------- 单图像推理模式 --------------------
if args.infer is not None:
    model = VNet().to(device)
    model.load_state_dict(torch.load(args.model,map_location=device))
    img = imread(args.infer)
    if img.ndim==3: img = img.mean(axis=2)
    img = img.astype(np.float32)
    img = img/np.max(np.abs(img))*2-1
    denoised = denoise_single_image(model,img)
    out_path = os.path.splitext(args.infer)[0]+'_denoised.png'
    imwrite(out_path,(denoised+1)/2*255)
    print(f"单张图像去噪完成，保存为 {out_path}")
    exit(0)

# -------------------- 加载训练数据集 --------------------
mat_path = 'C:/Users/25714/Documents/MATLAB/fringe_data_v3.mat'
data = sio.loadmat(mat_path)
X = torch.tensor(data['XTrain'],dtype=torch.float32).permute(3,2,0,1)
Y = torch.tensor(data['YTrain'],dtype=torch.float32).permute(3,2,0,1)

# -------------------- 划分训练/验证集 --------------------
n_samples = X.shape[0]
n_train = int(0.9*n_samples)
indices = np.random.permutation(n_samples)
train_idx,val_idx = indices[:n_train],indices[n_train:]
X_train,Y_train = X[train_idx],Y[train_idx]
X_val,Y_val = X[val_idx],Y[val_idx]

# -------------------- DataLoader --------------------
train_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=args.batch_size,shuffle=True)
val_loader = DataLoader(TensorDataset(X_val,Y_val),batch_size=args.batch_size,shuffle=False)

# -------------------- 初始化模型/优化器 --------------------
model = VNet().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=5)
criterion = nn.L1Loss()

# -------------------- 训练循环 --------------------
epochs = args.epochs
best_val_loss = float('inf')
train_losses,val_losses=[],[]

for epoch in range(epochs):
    # --- 训练 ---
    model.train()
    total_train_loss=0
    for batch_x,batch_y in train_loader:
        batch_x_aug,batch_y_aug = batch_x.clone(),batch_y.clone()
        for i in range(batch_x.size(0)):
            batch_x_aug[i],batch_y_aug[i] = augment(batch_x_aug[i],batch_y_aug[i])
        batch_x_aug,batch_y_aug = batch_x_aug.to(device),batch_y_aug.to(device)
        optimizer.zero_grad()
        pred = model(batch_x_aug)
        loss = criterion(pred,batch_y_aug)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss/len(train_loader)
    train_losses.append(avg_train_loss)

    # --- 验证 ---
    model.eval()
    total_val_loss=0
    with torch.no_grad():
        for batch_x,batch_y in val_loader:
            batch_x,batch_y = batch_x.to(device),batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred,batch_y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss/len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(),'best_vnet_fast.pth')
    if (epoch+1)%5==0:
        print(f"第{epoch+1}轮 | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f}")

# -------------------- 随机验证样本演示 --------------------
rand_idx = np.random.randint(0,len(X_val))
test_input_np = X_val[rand_idx].cpu().squeeze().numpy()
test_truth = Y_val[rand_idx].cpu().squeeze().numpy()
test_output_np = denoise_single_image(model,test_input_np)
psnr_val = 20*np.log10(2/np.sqrt(np.mean((test_output_np-test_truth)**2)))
print(f"随机验证样本 PSNR: {psnr_val:.2f} dB")

# -------------------- 可视化 --------------------
plt.figure(figsize=(15,10))
plt.subplot(2,3,1); plt.imshow(test_input_np,cmap='gray'); plt.title('含噪输入'); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(test_output_np,cmap='gray'); plt.title(f'V-net 去噪 (PSNR={psnr_val:.2f}dB)'); plt.axis('off')
plt.subplot(2,3,3); plt.imshow(test_truth,cmap='gray'); plt.title('干净相位图'); plt.axis('off')
plt.subplot(2,3,4); plt.plot(train_losses,label='训练损失'); plt.plot(val_losses,label='验证损失')
plt.legend(); plt.title('训练曲线'); plt.xlabel('轮次'); plt.ylabel('L1 损失'); plt.yscale('log')
plt.subplot(2,3,5); error_map = np.abs(test_output_np-test_truth); plt.imshow(error_map,cmap='hot'); plt.colorbar(); plt.title('绝对误差图')
plt.tight_layout()
plt.savefig('result_vnet_fast.png',dpi=150)
plt.show()
print("训练完成，随机验证样本演示完成，结果保存为 result_vnet_fast.png")
