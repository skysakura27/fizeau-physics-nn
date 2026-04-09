# 03-Algorithm-Unrolling.md

## Algorithm Unrolling Backbone

### 原理概述

算法展开（Algorithm Unrolling）将传统的迭代优化算法转换为深度神经网络：
- 每个迭代步对应一个网络层
- 学习算法参数而非手动调优
- 结合数据驱动和物理先验

### Fizeau 相位恢复算法

#### 传统迭代方法
1. **Gerchberg-Saxton 算法**: 幅度约束迭代
2. **Fienup 算法**: 支持约束迭代
3. **混合输入输出算法**: 逐步放松约束

#### 展开为神经网络
```python
class UnrolledPhaseRetrieval(nn.Module):
    def __init__(self, num_iterations=10):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnrollingBlock() for _ in range(num_iterations)
        ])

    def forward(self, interferogram):
        phase = self.initialize_phase(interferogram)
        for block in self.blocks:
            phase = block(phase, interferogram)
        return phase
```

### 展开块设计

#### 基本结构
每个展开块包含：
- **状态更新**: 基于当前相位估计
- **数据一致性**: 强制与观测数据匹配
- **物理约束**: 应用光学物理规律
- **正则化**: 防止过拟合

#### 学习参数
- **步长参数**: 每次迭代的学习率
- **正则化权重**: 不同约束的相对重要性
- **非线性激活**: 相位更新的非线性变换

### 与物理模型的集成

#### Airy 衍射模型
```python
def airy_psf(phase, wavelength, aperture):
    """
    Compute PSF using Airy diffraction model

    Args:
        phase: Phase distribution
        wavelength: Light wavelength
        aperture: Aperture function

    Returns:
        psf: Point spread function
    """
    # FFT-based diffraction calculation
    pass
```

#### 相位一致性约束
- **Helmholtz 方程**: 保证相位物理合理性
- **边界条件**: 孔径外的相位约束
- **连续性**: 相位空间平滑性

### 训练策略

#### 损失函数设计
```python
def physics_informed_loss(pred_phase, target_phase, interferogram):
    # 数据保真度损失
    data_loss = mse_loss(pred_phase, target_phase)

    # 物理一致性损失
    physics_loss = helmholtz_constraint(pred_phase)

    # 正则化损失
    reg_loss = smoothness_penalty(pred_phase)

    return data_loss + 0.1 * physics_loss + 0.01 * reg_loss
```

#### 端到端训练
- 使用真实干涉数据进行监督学习
- 结合无监督的物理约束
- 多尺度损失函数

### 优势

- **收敛速度**: 比传统方法快 10-100 倍
- **鲁棒性**: 对噪声和初始条件不敏感
- **泛化能力**: 学习到的参数适用于不同场景
- **可解释性**: 每个层对应物理意义明确的步骤