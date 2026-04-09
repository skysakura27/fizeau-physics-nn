# 04-Zernike-Polynomials.md

## Zernike Polynomials - Aberration Parameterization

### 数学基础

Zernike 多项式是正交基函数，用于描述光学系统的波前像差：

```
Z_n^m(ρ,θ) = R_n^m(ρ) * cos(mθ)    (m ≥ 0)
Z_n^{-m}(ρ,θ) = R_n^m(ρ) * sin(mθ)  (m > 0)
```

其中：
- `ρ`: 归一化径向坐标 (0 ≤ ρ ≤ 1)
- `θ`: 方位角 (0 ≤ θ < 2π)
- `R_n^m(ρ)`: 径向多项式
- `n`: 径向阶数
- `m`: 角阶数

### 标准 Zernike 模式

前 37 个 Zernike 模式对应常见的波前像差：

| 序号 | 名称 | 描述 |
|------|------|------|
| 1 | Piston | 活塞 |
| 2 | Tilt X | X 倾斜 |
| 3 | Tilt Y | Y 倾斜 |
| 4 | Defocus | 离焦 |
| 5 | Astigmatism 0° | 散光 0° |
| 6 | Astigmatism 45° | 散光 45° |
| ... | ... | ... |

### PyTorch 实现

#### Zernike 基函数生成
```python
import torch
import numpy as np

def zernike_basis(size=512, num_modes=37):
    """
    Generate Zernike polynomial basis

    Args:
        size: Image size
        num_modes: Number of Zernike modes

    Returns:
        basis: (num_modes, size, size) tensor
    """
    # Create coordinate grid
    y, x = torch.meshgrid(torch.linspace(-1, 1, size),
                         torch.linspace(-1, 1, size),
                         indexing='ij')

    rho = torch.sqrt(x**2 + y**2)
    theta = torch.atan2(y, x)

    # Mask for unit circle
    mask = rho <= 1

    basis = []
    for n, m in zernike_indices(num_modes):
        z = zernike_polynomial(n, m, rho, theta)
        z = z * mask
        basis.append(z)

    return torch.stack(basis)
```

#### 系数投影
```python
def project_to_zernike(phase, basis):
    """
    Project phase onto Zernike basis

    Args:
        phase: Phase map (H, W)
        basis: Zernike basis (num_modes, H, W)

    Returns:
        coeffs: Zernike coefficients (num_modes,)
    """
    # Inner product with basis functions
    coeffs = torch.sum(phase.unsqueeze(0) * basis, dim=[1, 2])
    coeffs = coeffs / torch.sum(basis**2, dim=[1, 2])
    return coeffs
```

### 监督学习集成

#### Zernike 监督器
```python
class ZernikeSupervisor(nn.Module):
    def __init__(self, size=512, num_modes=37):
        super().__init__()
        self.basis = zernike_basis(size, num_modes)
        self.num_modes = num_modes

    def forward(self, phase):
        """
        Extract Zernike coefficients from phase

        Args:
            phase: Predicted phase (batch, 1, H, W)

        Returns:
            coeffs: Zernike coefficients (batch, num_modes)
        """
        batch_size = phase.shape[0]
        coeffs = []

        for b in range(batch_size):
            coeff = project_to_zernike(phase[b, 0], self.basis)
            coeffs.append(coeff)

        return torch.stack(coeffs)
```

#### 系数锁定机制
```python
def lock_coefficients(coeffs, locked_modes):
    """
    Lock specific Zernike coefficients to known values

    Args:
        coeffs: Current coefficients
        locked_modes: Dict of {mode_idx: target_value}

    Returns:
        locked_coeffs: Modified coefficients
    """
    locked_coeffs = coeffs.clone()
    for mode_idx, value in locked_modes.items():
        locked_coeffs[:, mode_idx] = value
    return locked_coeffs
```

### 应用场景

#### 光学系统校正
- **自适应光学**: 实时波前校正
- **镜头设计**: 优化光学元件
- **质量控制**: 检测制造偏差

#### 相位恢复中的作用
- **参数化约束**: 将无限维相位空间投影到有限维
- **物理意义**: 每个系数对应特定类型的像差
- **正则化**: 防止过拟合和不合理相位分布

### 数值稳定性

- **正交性**: 保证系数独立性
- **归一化**: 统一系数尺度
- **边界处理**: 正确处理单位圆边界

### 扩展应用

- **高阶像差**: 扩展到更高阶 Zernike 模式
- **非圆孔径**: 适应不同形状的光瞳
- **动态校正**: 时变像差的跟踪