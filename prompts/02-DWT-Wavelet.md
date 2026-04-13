# 02-DWT-Wavelet.md

## DWT (Discrete Wavelet Transform) Preprocessing Layer

### 功能概述

2D-DWT 层用于对输入干涉图进行多尺度小波变换，实现：
- 噪声抑制和特征增强
- 多分辨率特征提取
- 相位信息的尺度不变性处理

### PyTorch 模块示例（固定 2D Haar）
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTPreprocessLayer(nn.Module):
    """
    Fixed 2D Haar DWT preprocessing layer.

    输入: (B, 1, H, W)
    输出: (B, 4, H/2, W/2), 顺序: LL, LH, HL, HH
    """

    def __init__(self):
        super().__init__()
        weight = self._build_haar_kernels()
        self.register_buffer("haar_weight", weight)

    @staticmethod
    def _build_haar_kernels() -> torch.Tensor:
        low = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
        high = torch.tensor([-1.0, 1.0]) / math.sqrt(2.0)

        ll = torch.ger(low, low)
        lh = torch.ger(low, high)
        hl = torch.ger(high, low)
        hh = torch.ger(high, high)

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("DWTPreprocessLayer expects input shape (B, 1, H, W).")
        if x.size(-1) % 2 != 0 or x.size(-2) % 2 != 0:
            raise ValueError("Input H and W must be even for 2x2 Haar DWT.")

        weight = self.haar_weight.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, weight, stride=2)
```

### 为什么有助于分离条纹与散斑

干涉条纹主要体现为低频、平滑且具有方向性的周期结构，而散斑噪声通常更高频、局部随机。2D Haar DWT 将能量分解到不同子带：
- **LL**: 低频近似，保留主要条纹和整体光强变化
- **LH/HL**: 中高频方向细节，突出条纹边缘与方向结构
- **HH**: 高频细节，包含较多散斑噪声

通过在网络中对不同子带进行针对性建模或权重融合，可以增强条纹成分、抑制散斑噪声，从而实现更清晰的物理相位恢复与去噪效果。

### 实现要点

#### 小波基选择
- **Haar 小波**: 计算简单，适合快速处理
- **Daubechies 小波**: 更好的频率特性
- **Symlet 小波**: 对称性更好，减少边界效应

#### 多尺度分解
```python
def wavelet_decomposition(x, levels=3):
    """
    Perform 2D wavelet decomposition

    Args:
        x: Input tensor (batch, 1, H, W)
        levels: Decomposition levels

    Returns:
        coeffs: List of approximation and detail coefficients
    """
    # Implementation using PyWavelets or custom DWT
    pass
```

#### 逆变换重建
```python
def wavelet_reconstruction(coeffs):
    """
    Reconstruct signal from wavelet coefficients

    Args:
        coeffs: Wavelet coefficients

    Returns:
        reconstructed: Reconstructed signal
    """
    pass
```

### 与深度学习的集成

#### 特征融合策略
1. **Concatenation**: 将不同尺度系数拼接
2. **Attention Mechanism**: 学习尺度重要性权重
3. **Progressive Fusion**: 逐层融合多尺度信息

#### 梯度流优化
- 确保 DWT 层可微分
- 使用自定义 autograd 函数实现逆变换
- 内存高效的系数存储

### 应用场景

- **噪声去除**: 小波域的阈值去噪
- **特征增强**: 多尺度边缘检测
- **相位平滑**: 不同尺度的相位一致性约束

### 参考实现

基于 `pytorch-wavelets` 库或自定义 CUDA 实现，确保：
- 支持任意输入尺寸
- 高效的 GPU 计算
- 与 PyTorch 自动求导兼容
