# 05-Residual-Denoising.md

## Residual Denoising Networks

### 原理概述

残差去噪网络（Residual Denoising Networks）在每个算法展开块中加入：
- 学习残差噪声模式
- 去除系统性和随机噪声
- 提高相位重建精度

### 网络架构

#### 基本结构
```python
class ResidualDenoisingBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

        # Residual connection
        self.skip = nn.Conv2d(1, 1, 1)  # 1x1 conv for channel adjustment

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Decoder with residual
        residual = self.decoder(features)
        skip = self.skip(x)

        return residual + skip
```

#### 多尺度设计
```python
class MultiScaleDenoising(nn.Module):
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.blocks = nn.ModuleList([
            ResidualDenoisingBlock() for _ in scales
        ])

    def forward(self, x):
        outputs = []
        for scale, block in zip(self.scales, self.blocks):
            # Downsample
            if scale > 1:
                x_scaled = F.interpolate(x, scale_factor=1/scale, mode='bilinear')
            else:
                x_scaled = x

            # Denoise
            denoised = block(x_scaled)

            # Upsample back
            if scale > 1:
                denoised = F.interpolate(denoised, scale_factor=scale, mode='bilinear')

            outputs.append(denoised)

        # Fuse multi-scale outputs
        return self.fuse_outputs(outputs)
```

### 训练策略

#### 噪声模拟
```python
def add_synthetic_noise(clean_phase, snr_db=20):
    """
    Add realistic noise to clean phase data

    Args:
        clean_phase: Ground truth phase
        snr_db: Signal-to-noise ratio in dB

    Returns:
        noisy_phase: Phase with added noise
        noise_pattern: Noise component for training
    """
    # Shot noise
    shot_noise = torch.poisson(clean_phase.abs()**2) - clean_phase.abs()**2

    # Readout noise
    readout_noise = torch.randn_like(clean_phase) * 0.01

    # Quantization noise
    quant_noise = torch.round(clean_phase * 100) / 100 - clean_phase

    # Combine noises
    total_noise = shot_noise + readout_noise + quant_noise

    # Scale to desired SNR
    noise_power = torch.var(total_noise)
    signal_power = torch.var(clean_phase)
    target_noise_power = signal_power / (10**(snr_db/10))

    if noise_power > 0:
        scale = torch.sqrt(target_noise_power / noise_power)
        total_noise = total_noise * scale

    return clean_phase + total_noise, total_noise
```

#### 损失函数
```python
def denoising_loss(pred_noise, true_noise, pred_clean, true_clean):
    """
    Combined loss for denoising task

    Args:
        pred_noise: Predicted noise component
        true_noise: Ground truth noise
        pred_clean: Predicted clean signal
        true_clean: Ground truth clean signal

    Returns:
        total_loss: Combined loss value
    """
    # Noise prediction loss
    noise_loss = F.mse_loss(pred_noise, true_noise)

    # Reconstruction loss
    recon_loss = F.mse_loss(pred_clean, true_clean)

    # Consistency loss
    consistency_loss = F.mse_loss(pred_clean - pred_noise, true_clean - true_noise)

    return noise_loss + recon_loss + 0.1 * consistency_loss
```

### 与算法展开的集成

#### 展开块中的去噪
```python
class UnrollingBlockWithDenoising(nn.Module):
    def __init__(self):
        super().__init__()
        self.physics_update = PhysicsUpdateLayer()
        self.denoising_net = ResidualDenoisingBlock()
        self.adaptive_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, phase, interferogram):
        # Physics-based update
        phase_phys = self.physics_update(phase, interferogram)

        # Residual denoising
        residual = self.denoising_net(phase_phys - phase)
        phase_denoised = phase_phys + self.adaptive_weight * residual

        return phase_denoised
```

### 性能优化

#### 内存效率
- **渐进式训练**: 从低分辨率开始
- **梯度累积**: 减少显存占用
- **混合精度**: 使用 FP16 训练

#### 计算效率
- **深度可分离卷积**: 减少参数量
- **注意力机制**: 聚焦重要区域
- **知识蒸馏**: 从大模型到小模型

### 评估指标

#### 定量指标
- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性指数
- **RMSE**: 均方根误差

#### 定性评估
- **视觉检查**: 相位图的连续性和合理性
- **频谱分析**: 噪声在频域的分布
- **收敛性**: 迭代过程中的误差下降曲线

### 实际应用

- **低信噪比场景**: 增强弱信号检测
- **实时处理**: 快速去噪算法
- **自适应校正**: 根据噪声类型调整策略