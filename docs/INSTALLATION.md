# INSTALLATION.md

# 安装指南

## 系统要求

### 最低系统要求

- **操作系统**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python**: 3.8 - 3.11
- **内存**: 8GB RAM (推荐 16GB+)
- **存储**: 5GB 可用空间
- **GPU**: NVIDIA GPU with CUDA 11.0+ (可选, 推荐用于训练)

### 推荐配置

- **CPU**: Intel i7/AMD Ryzen 7 或更高
- **GPU**: NVIDIA RTX 3060 或更高 (8GB+ VRAM)
- **内存**: 32GB RAM
- **存储**: SSD with 50GB+ 可用空间

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/fizeau-physics-nn.git
cd fizeau-physics-nn
```

### 2. 创建虚拟环境

#### 使用 conda (推荐)

```bash
# 创建环境
conda create -n fizeau-pinn python=3.9
conda activate fizeau-pinn

# 或者使用 mamba (更快)
mamba create -n fizeau-pinn python=3.9
mamba activate fizeau-pinn
```

#### 使用 venv

```bash
# 创建环境
python -m venv fizeau-env

# Windows
fizeau-env\Scripts\activate

# Linux/macOS
source fizeau-env/bin/activate
```

### 3. 安装依赖

#### 基本安装

```bash
pip install -r requirements.txt
```

#### 开发安装 (包含测试和文档工具)

```bash
pip install -r requirements-dev.txt
```

#### 可选依赖

```bash
# GPU 支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 额外科学计算库
pip install scipy scikit-image

# 可视化增强
pip install plotly bokeh
```

## 依赖说明

### 核心依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| torch | >=1.12.0 | 深度学习框架 |
| torchvision | >=0.13.0 | 计算机视觉组件 |
| numpy | >=1.21.0 | 数值计算 |
| scipy | >=1.7.0 | 科学计算 |
| PyYAML | >=6.0 | 配置管理 |
| pathlib | - | 路径处理 (Python 3.4+) |
| matplotlib | >=3.5.0 | 基础绘图 |
| tqdm | >=4.62.0 | 进度条 |

### 开发依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| pytest | >=7.0.0 | 单元测试 |
| pytest-cov | >=3.0.0 | 测试覆盖率 |
| black | >=22.0.0 | 代码格式化 |
| flake8 | >=4.0.0 | 代码检查 |
| mypy | >=0.950 | 类型检查 |
| sphinx | >=4.5.0 | 文档生成 |
| jupyter | >=1.0.0 | 笔记本支持 |

## GPU 配置

### CUDA 安装

#### Windows

1. 下载 CUDA Toolkit 11.8: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. 安装 CUDA 和 cuDNN
3. 验证安装:

```bash
nvcc --version
nvidia-smi
```

#### Linux

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

### PyTorch GPU 版本

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 验证 GPU 可用性

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current GPU: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

## 配置设置

### 环境变量

```bash
# 设置 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# CUDA 相关 (如果需要)
export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1
```

### 配置文件

复制并修改配置模板:

```bash
cp configs/model_cfg.yaml configs/model_cfg.local.yaml
cp configs/physics_cfg.yaml configs/physics_cfg.local.yaml
# 编辑 *.local.yaml 根据需要修改参数
```

### IDE 配置

#### VS Code

安装推荐扩展:
- Python
- Pylance
- Jupyter
- GitLens

.vscode/settings.json 已包含项目配置。

#### PyCharm

- 设置项目解释器为 conda 环境
- 启用科学模式
- 配置运行/调试配置

## 验证安装

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_physics_models.py

# 带覆盖率
pytest --cov=src --cov-report=html
```

### 运行示例

```bash
# 运行最小示例
python examples/minimal-example.py

# 检查输出是否正常
```

### 性能测试

```bash
# GPU 性能测试
python -c "import torch; print(torch.cuda.is_available())"

# 内存测试
python -c "import src.models.unrolled_net; print('Import successful')"
```

## 故障排除

### 常见问题

#### 1. CUDA 版本不匹配

```
RuntimeError: CUDA error: no kernel image is available
```

**解决方案**:
- 检查 CUDA 版本: `nvcc --version`
- 安装匹配的 PyTorch 版本
- 更新 GPU 驱动

#### 2. 内存不足

```
CUDA out of memory
```

**解决方案**:
- 减小批次大小
- 使用梯度累积
- 启用混合精度训练

#### 3. 依赖冲突

```
ImportError: No module named 'xxx'
```

**解决方案**:
- 重新创建虚拟环境
- 检查 requirements.txt
- 手动安装缺失包

#### 4. 编译错误

```
error: command 'gcc' failed
```

**解决方案** (Linux):
```bash
sudo apt-get update
sudo apt-get install build-essential
```

### 调试步骤

1. **检查 Python 版本**:
   ```bash
   python --version
   ```

2. **检查包版本**:
   ```bash
   pip list | grep torch
   ```

3. **检查 GPU 状态**:
   ```bash
   nvidia-smi
   ```

4. **运行诊断脚本**:
   ```python
   import sys
   print("Python path:", sys.path)
   import torch
   print("Torch version:", torch.__version__)
   print("CUDA available:", torch.cuda.is_available())
   ```

## 更新和维护

### 更新依赖

```bash
# 更新所有包
pip install --upgrade -r requirements.txt

# 检查过时包
pip list --outdated
```

### 重新安装

```bash
# 完全重新安装
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

## Docker 支持

### 构建 Docker 镜像

```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "examples/minimal-example.py"]
```

```bash
# 构建镜像
docker build -t fizeau-pinn .

# 运行容器
docker run --gpus all fizeau-pinn
```

## 贡献者指南

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install

# 运行代码质量检查
black src tests
flake8 src tests
mypy src
```

### 测试要求

- 所有测试必须通过: `pytest`
- 代码覆盖率 > 80%
- 无 flake8 错误
- 类型检查通过

## 支持

### 获取帮助

- **问题**: 在 GitHub Issues 中报告
- **讨论**: 使用 GitHub Discussions
- **文档**: 查看 docs/ 目录

### 系统信息收集

运行以下命令收集系统信息:

```bash
python -c "
import platform, torch, sys
print('OS:', platform.platform())
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name() if torch.cuda.is_available() else 'None')
"
```
