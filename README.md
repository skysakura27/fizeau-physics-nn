# Fizeau Physics-Informed Neural Network (PINN)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/your-username/fizeau-physics-nn/workflows/Tests/badge.svg)](https://github.com/your-username/fizeau-physics-nn/actions)

**突破传统干涉解相算法与黑盒深度学习的瓶颈，实现 Fizeau 干涉仪 2nm 级别的超高精度、强鲁棒性波前测量。**

基于物理信息神经网络的 Fizeau 干涉仪相位重建算法，采用"由表及里"的物理闭环技术路径：
- **第一阶段**：物理信号获取与解构 - 分析干涉图中的信号与核心噪声源（散斑噪声、寄生条纹）
- **第二阶段**：频域解耦预处理 - 2D-DWT 通道分离，实现信号与散斑的解耦
- **第三阶段**：物理嵌入式核心引擎 - Airy-PIN 迭代层（物理纠偏 + 残差清洗）
- **第四阶段**：拓扑硬约束输出 - Zernike 正交基面型硬约束

## ✨ 核心特性

- **🎯 2nm 级超高精度**: 通过物理闭环约束实现纳米级波前测量重复性
- **🔬 物理信号解构**: 深入分析 Fizeau 干涉图的信号生成机理和噪声机理
- **🌊 频域解耦预处理**: 2D-DWT 抑制散斑噪声，实现信号与噪声的频域分离
- **🌀 Airy-PIN 迭代层**: 算法展开架构，将多光束干涉物理法则嵌入网络前向传播
- **📐 Zernike 硬约束**: 36 阶正交多项式基底保证输出波前的物理拓扑结构
- **🚀 强鲁棒性**: 杜绝纯数据驱动的泛化灾难，适用于各种噪声条件

## 📋 目录

- [快速开始](#-快速开始)
- [安装](#-安装)
- [使用方法](#-使用方法)
- [技术路径详解](#-技术路径详解)
- [架构详解](#-架构详解)
- [训练](#-训练)
- [评估](#-评估)
- [API 文档](#-api-文档)
- [贡献](#-贡献)
- [引用](#-引用)
- [许可证](#-许可证)

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于 GPU 加速)

### 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/fizeau-physics-nn.git
cd fizeau-physics-nn

# 创建虚拟环境
conda create -n fizeau-pinn python=3.9
conda activate fizeau-pinn

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

```python
import torch
from src.models import FizeauPhysicsNet
from src.utils import Config

# 加载配置
config = Config()

# 创建模型
model = FizeauPhysicsNet(config.get("network", {}))

# 生成测试数据
interferogram = torch.rand(1, 1, 512, 512)  # 干涉图

# 推理
with torch.no_grad():
    phase = model(interferogram)

print(f"重建相位形状: {phase.shape}")
```

## � 技术路径详解

### 第一阶段：物理信号的获取与解构（问题定义）

**核心任务**：分析 Fizeau 干涉图中的信号与核心噪声源。

**物理原理**：
- **信号生成 (Fizeau 干涉)**：经典共路干涉。测试光与参考光干涉，理论上遵循两光束的正弦干涉规律。但由于高反射率表面的多光束干涉效应，实际光强分布呈现非线性的 Airy 函数形态：$I = \frac{I_{max}}{1 + F \sin^2(\phi/2)}$
- **噪声机理**：
  - **散斑噪声 (Speckle)**：相干光照射粗糙表面产生的随机干涉，在频域上表现为高频特征
  - **寄生条纹 (牛眼环)**：系统内部非工作面的多重反射，在频域上表现为与主条纹交织的中频空间结构

### 第二阶段：频域解耦预处理（The Gatekeeper）

**核心任务**：引入二维离散小波变换（2D-DWT）对输入含噪干涉图进行通道分离。

**物理原理 (频域滤波)**：利用小波变换的时频局部化特性。将图像分解为低频（轮廓）和高频（细节）分量。

**物理动作**：识别并抑制高频对角分量（HH，即散斑的主要藏身处），提取低频和中频分量（LL, LH, HL）。

**意义**：在进入深度网络前，遵循物理频域特征完成"信号与散斑"的解耦，大幅降低神经网络提取特征的难度，避免网络在拟合高频噪声上浪费算力。

### 第三阶段：物理嵌入式核心引擎（The Unrolled Engine）

**核心任务**：摒弃"端到端"的黑盒 U-Net，采用算法展开 (Algorithm Unrolling) 架构，构建"Airy-PIN 迭代层"。

**物理原理 (近端梯度下降与多光束干涉)**：将神经网络的层级结构映射为物理方程的迭代求解过程。

**每一层执行两个物理动作**：
- **动作 A（物理纠偏）**：硬编码 Airy 函数梯度算子 $\nabla_{\phi} \mathcal{L}_{Airy}$。根据多光束干涉的物理法则，计算当前相位猜测值带来的强度残差，并强制相位向符合物理光强分布的方向更新
- **动作 B（残差清洗）**：利用轻量级 CNN 充当近端算子 (Proximal Operator)。由于物理公式无法描述系统中的随机热噪声和未知寄生环，CNN 负责吸收并剔除这些脱离 Airy 模型的"非物理残差"

**意义**：实现了"架构即物理"，保证相位解调的每一步都在物理梯度的指引下进行，杜绝了纯数据驱动带来的泛化灾难。

### 第四阶段：拓扑硬约束输出（The Zernike Supervisor）

**核心任务**：网络末端放弃逐像素（Pixel-wise）相位输出，改为预测 36 阶 Zernike 多项式系数，最后通过矩阵乘法还原三维相位。

**物理原理 (像差正交基底投影)**：Zernike 多项式是在单位圆上正交的连续多项式集，是光学界描述波前像差（如离焦、像散、彗差）的"官方语言"和物理基底。

**物理动作**：这是一个硬约束 (Hard Constraint)。强制网络预测的相位必须是由这 36 种标准物理面型线性叠加而成。

**意义**：从数学拓扑结构上彻底封死了网络输出断裂、突变或非物理高频抖动的可能性，锁定波前的纳米级"宏观骨架"。

**总结**："我们的技术路径是一条'由表及里'的物理闭环：在输入端用 DWT 解决频域混叠，在计算核心用算法展开将 Airy 非线性模型植入网络前向传播，最后在输出端用 Zernike 正交基进行面型硬约束。这套方案用物理法则框定了神经网络的自由度，是我们实现 Fizeau 干涉仪 2nm 测量重复性的核心壁垒。"

## �📦 安装

详细安装说明请参考 [INSTALLATION.md](docs/INSTALLATION.md)

### 开发安装

```bash
pip install -r requirements-dev.txt
```

## 💡 使用方法

### 基本使用

```python
from src.models import FizeauPhysicsNet
from src.utils.data_loader import InterferometryDataset
from torch.utils.data import DataLoader

# 加载模型
model = FizeauPhysicsNet.load_from_checkpoint('checkpoints/best_model.ckpt')

# 准备数据
dataset = InterferometryDataset('data/test/')
loader = DataLoader(dataset, batch_size=4)

# 批量推理
for batch in loader:
    interferogram = batch['interferogram']
    with torch.no_grad():
        phase_pred = model(interferogram)

    # 保存结果
    save_phase_map(phase_pred, batch['filename'])
```

### 高级配置

```python
from src.utils import Config

# 自定义配置
config = Config()
config.model["network"]["num_unrolling_blocks"] = 8
config.model["network"]["zernike_modes"] = 55
config.model["training"]["learning_rate"] = 1e-4

# 创建模型
model = FizeauPhysicsNet(config.get("network", {}))
```

## 🏗️ 架构详解

### 网络结构

```
输入干涉图 → DWT预处理 → 算法展开网络 → Zernike监督 → 输出相位
                    ↓
              残差去噪分支
```

### 核心组件

1. **DWT 预处理层**
   - 2D Haar 小波分解
   - 多尺度特征提取
   - 噪声鲁棒性增强

2. **算法展开网络**
   - N 个展开块 (N=5-10)
   - 每个块：物理更新 + 神经校正
   - 自适应权重学习

3. **残差去噪网络**
   - U-Net 架构
   - 多尺度去噪
   - 注意力机制

4. **Zernike 监督器**
   - 37/55 阶 Zernike 模态
   - 物理可解释性
   - 正则化约束

### 技术规格

| 组件 | 参数 | 说明 |
|------|------|------|
| 输入分辨率 | 512×512 | 可配置 |
| Zernike 模态 | 37/55 | 标准/扩展 |
| 展开层数 | 5-10 | 可调 |
| 网络参数 | ~5M | 轻量级 |

## 🎯 训练

### 数据准备

```bash
# 生成合成训练数据
python scripts/generate_training_data.py \
    --num_samples 10000 \
    --image_size 512 \
    --snr_range 10 40 \
    --output_dir data/train/
```

### 训练模型

```bash
# 单 GPU 训练
python scripts/train.py \
    --model_cfg configs/model_cfg.yaml \
    --physics_cfg configs/physics_cfg.yaml \
    --data_dir data/train/ \
    --output_dir checkpoints/

# 多 GPU 训练
torchrun --nproc_per_node=4 scripts/train.py \
    --model_cfg configs/model_cfg.yaml \
    --physics_cfg configs/physics_cfg.yaml \
    --data_dir data/train/ \
    --output_dir checkpoints/
```

### 训练监控

```bash
# 使用 TensorBoard
tensorboard --logdir logs/

# 或使用 Weights & Biases
wandb login
python scripts/train.py --use_wandb
```

## 📊 评估

### 性能指标

- **PSNR**: 峰值信噪比
- **SSIM**: 结构相似性
- **RMSE**: 均方根误差
- **PV/RMS**: 表面精度

### 运行评估

```bash
# 评估模型
python scripts/evaluate.py \
    --model_path checkpoints/best_model.ckpt \
    --data_dir data/test/ \
    --output_dir results/

# 生成评估报告
python scripts/generate_report.py \
    --results_dir results/ \
    --output_file evaluation_report.pdf
```

## 📚 API 文档

### 核心类

#### `FizeauPhysicsNet`

主网络类，实现完整的 PINN 架构。

```python
class FizeauPhysicsNet(nn.Module):
    def __init__(self, config: Dict):
        """初始化网络

        Args:
            config: 网络配置字典
        """

    def forward(self, interferogram: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            interferogram: 输入干涉图 [B, 1, H, W]

        Returns:
            phase: 重建相位 [B, 1, H, W]
        """
```

#### `Config`

配置管理类。

```python
class Config:
    def __init__(self, model_path="configs/model_cfg.yaml", physics_path="configs/physics_cfg.yaml"):
        """加载模型与物理配置"""

    def get(self, key: str, default=None):
        """读取配置项"""
```

### 工具函数

#### 物理模型

```python
from src.core import AiryPhysicsModel, ZernikeBasis

# Airy 前向模型（示例）
physics = AiryPhysicsModel({"wavelength": 632e-9, "finesse": 10})
interferogram = physics.forward_model(phase_map.unsqueeze(0).unsqueeze(0))

# Zernike 基函数
zernike = ZernikeBasis(num_modes=37)
coeffs = zernike.project(phase_map)
reconstructed = zernike.reconstruct(coeffs)
```

#### 数据处理

```python
from src.utils.data_loader import InterferometryDataset

# 创建数据集
dataset = InterferometryDataset(
    data_dir='data/',
    transform=Compose([
        Normalize(),
        AddNoise(snr_db=20),
        RandomCrop(512)
    ])
)
```

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 开发指南

- 遵循 PEP 8 代码风格
- 添加类型注解
- 编写单元测试
- 更新文档

### 测试

```bash
# 运行所有测试
pytest

# 带覆盖率
pytest --cov=src --cov-report=html

# 运行特定测试
pytest tests/test_physics_models.py::TestAiryPSF::test_psf_normalization
```

## 📖 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@article{fizeau-pinn-2024,
  title={Physics-Informed Neural Network for Fizeau Interferometry Phase Retrieval},
  author={Your Name},
  journal={arXiv preprint},
  year={2024},
  url={https://github.com/your-username/fizeau-physics-nn}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 感谢 PyTorch 团队提供的优秀深度学习框架
- 感谢光学社区对干涉测量技术的贡献
- 感谢所有贡献者的宝贵建议和代码改进

## 📞 联系

- **作者**: Your Name
- **邮箱**: your.email@example.com
- **GitHub**: [@your-username](https://github.com/your-username)
- **项目主页**: [https://github.com/your-username/fizeau-physics-nn](https://github.com/your-username/fizeau-physics-nn)

---

⭐ 如果这个项目对您有帮助，请给它一个星标！
