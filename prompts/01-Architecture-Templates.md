# 01-Architecture-Templates.md

## Fizeau Physics-Informed Neural Network - Architecture Templates

### 四阶段技术路径架构总览

```
第一阶段：物理信号的获取与解构
├── 信号生成机理分析 (Fizeau干涉)
├── 噪声机理分析 (散斑 + 寄生条纹)
└── 问题定义与解耦策略

第二阶段：频域解耦预处理 (The Gatekeeper)
├── Input: Interferogram (512x512)
├── 2D-DWT 通道分离
│   ├── 抑制HH分量 (散斑噪声)
│   ├── 保留LL/LH/HL分量 (有用信号)
│   └── 频域特征解耦
└── Output: 解耦后的多尺度特征

第三阶段：物理嵌入式核心引擎 (The Unrolled Engine)
├── Airy-PIN 迭代层 (5-10层)
│   ├── 动作A：物理纠偏 ∇_φℒ_Airy
│   ├── 动作B：残差清洗 CNN近端算子
│   └── 架构即物理保证
├── 残差去噪网络 (并行分支)
│   ├── U-Net架构
│   ├── 多尺度去噪
│   └── 注意力机制
└── Output: 物理约束的相位估计

第四阶段：拓扑硬约束输出 (The Zernike Supervisor)
├── Input: 相位估计 (512x512)
├── Zernike多项式基底 (36阶)
│   ├── 硬约束投影
│   ├── 系数预测网络
│   └── 面型重建
└── Output: Zernike系数 (36维) + 重建相位
```

### 核心组件接口

#### 第一阶段：物理信号分析
```python
class PhysicalSignalAnalyzer:
    """
    分析干涉图中的信号与噪声机理
    """
    def analyze_signal_generation(self, interferogram):
        """信号生成机理：I = I_max / (1 + F*sin²(φ/2))"""
        pass

    def analyze_noise_sources(self, interferogram):
        """噪声机理：散斑噪声 + 寄生条纹"""
        pass
```

#### 第二阶段：DWT频域解耦预处理器
```python
class DWTPreprocessor(nn.Module):
    """
    频域解耦预处理 - The Gatekeeper
    """
    def __init__(self, wavelet='haar', levels=2):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        # Haar滤波器实现

    def forward(self, interferogram):
        """
        2D-DWT分解，实现信号与噪声的频域分离

        Args:
            interferogram: 输入干涉图 (B, 1, H, W)

        Returns:
            {
                'LL': 低频近似系数 (有用信号),
                'LH': 水平细节,
                'HL': 垂直细节,
                'HH': 对角细节 (散斑噪声 - 被抑制)
            }
        """
        # 抑制HH分量，保留其他分量
        return decoupled_features
```

#### 第三阶段：Airy-PIN迭代层
```python
class AiryPINBlock(nn.Module):
    """
    Airy Physics-Informed Neural Network Block
    每个块执行两个物理动作
    """
    def __init__(self, config):
        super().__init__()
        self.physics_model = AiryPhysicsModel(config)
        self.proximal_net = nn.Sequential(...)  # CNN近端算子
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 自适应权重

    def forward(self, phase, interferogram):
        """
        Airy-PIN块前向传播

        Args:
            phase: 当前相位估计 (B, 1, H, W)
            interferogram: 测量干涉图 (B, 1, H, W)

        Returns:
            phase_updated: 更新后的相位
            info: 中间结果字典
        """
        # 动作A：物理纠偏
        physics_correction = self.physics_model.compute_gradient(phase, interferogram)

        # 动作B：残差清洗
        neural_correction = self.proximal_net(phase)

        # 自适应组合
        total_correction = self.alpha * physics_correction + (1 - self.alpha) * neural_correction

        return phase + total_correction, {
            'physics_correction': physics_correction,
            'neural_correction': neural_correction,
            'alpha': self.alpha
        }
```

#### 第三阶段：残差去噪网络
```python
class ResidualDenoisingNet(nn.Module):
    """
    残差去噪网络 - 抑制散斑噪声
    """
    def __init__(self, config):
        super().__init__()
        # U-Net架构实现
        # 编解码器 + 跳跃连接

    def forward(self, phase):
        """
        多尺度残差去噪

        Args:
            phase: 输入相位 (B, 1, H, W)

        Returns:
            denoised_phase: 去噪后的相位
        """
        return denoised_phase
```

#### 第四阶段：Zernike监督器
```python
class ZernikeSupervisor(nn.Module):
    """
    Zernike监督器 - 拓扑硬约束输出
    """
    def __init__(self, config):
        super().__init__()
        self.zernike_basis = ZernikeBasis(num_modes=36)
        self.coeff_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 36)  # 36阶Zernike系数
        )

    def forward(self, phase):
        """
        Zernike硬约束输出

        Args:
            phase: 输入相位图 (B, 1, H, W)

        Returns:
            reconstructed_phase: 从系数重建的相位 (B, 1, H, W)
            coefficients: Zernike系数 (B, 36)
        """
        # 预测系数
        coefficients = self.coeff_predictor(phase)

        # 重建相位 (硬约束)
        reconstructed_phase = self.zernike_basis.reconstruct(coefficients)

        return reconstructed_phase, coefficients
```

### 完整网络架构
```python
class FizeauPhysicsNet(nn.Module):
    """
    完整四阶段 Fizeau PINN
    """
    def __init__(self, config):
        super().__init__()

        # 第二阶段：频域解耦
        self.dwt_preprocessor = DWTPreprocessor(config['dwt'])

        # 第三阶段：Airy-PIN迭代层
        self.unrolling_blocks = nn.ModuleList([
            AiryPINBlock(config['block']) for _ in range(config['num_blocks'])
        ])

        # 第三阶段：残差去噪 (并行)
        self.denoising_net = ResidualDenoisingNet(config['denoising'])

        # 第四阶段：Zernike硬约束
        self.zernike_supervisor = ZernikeSupervisor(config['zernike'])

    def forward(self, interferogram):
        """
        四阶段前向传播

        Args:
            interferogram: 输入干涉图 (B, 1, H, W)

        Returns:
            final_phase: 最终重建相位 (B, 1, H, W)
            outputs: 各阶段输出字典
        """
        # 第一阶段：物理信号分析 (在预处理中体现)

        # 第二阶段：频域解耦
        dwt_coeffs = self.dwt_preprocessor(interferogram)

        # 第三阶段：物理嵌入式核心引擎
        phase = dwt_coeffs['LL']  # 初始相位估计
        for block in self.unrolling_blocks:
            denoised = self.denoising_net(phase)
            phase, _ = block(phase, interferogram)

        # 第四阶段：拓扑硬约束输出
        final_phase, zernike_coeffs = self.zernike_supervisor(phase)

        return final_phase, {
            'dwt_coeffs': dwt_coeffs,
            'zernike_coeffs': zernike_coeffs,
            'intermediate_phase': phase
        }
```

### 配置模板
```python
config = {
    'dwt': {
        'wavelet': 'haar',
        'levels': 2
    },
    'block': {
        'channels': 64,
        'physics': {
            'wavelength': 632.8e-9,
            'finesse': 10
        }
    },
    'num_blocks': 5,
    'denoising': {
        'channels': 64
    },
    'zernike': {
        'num_modes': 36
    }
}
```
```

#### Zernike Supervisor
```python
class ZernikeSupervisor(nn.Module):
    def __init__(self, num_modes=37):
        # Zernike basis functions
        # Coefficient estimation

    def forward(self, phase):
        # Project phase onto Zernike basis
        return zernike_coeffs
```

### 训练策略

1. **Multi-task Learning**: 同时优化相位重建和 Zernike 系数
2. **Physics Constraints**: 集成 Airy 衍射模型
3. **Regularization**: 相位平滑性和稀疏性约束
4. **Curriculum Learning**: 从简单到复杂的干涉图训练

### 评估指标

- **RMS Wavefront Error**: 相位重建精度
- **Zernike Coefficient Accuracy**: 像差参数化精度
- **PSNR/SSIM**: 图像质量指标
- **Computational Efficiency**: 推理时间和内存使用