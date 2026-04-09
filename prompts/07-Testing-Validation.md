# 07-Testing-Validation.md

## Testing and Validation Framework

### 测试架构

#### 目录结构
```
tests/
├── __init__.py
├── conftest.py                    # pytest 配置
├── test_physics_models.py         # 物理模型测试
├── test_network_components.py     # 网络组件测试
├── test_training_pipeline.py      # 训练流程测试
├── test_data_pipeline.py          # 数据管道测试
├── test_integration.py            # 集成测试
├── fixtures/
│   ├── sample_data.py            # 测试数据生成
│   └── mock_models.py            # 模拟模型
└── benchmarks/
    ├── performance_benchmarks.py # 性能基准测试
    └── accuracy_benchmarks.py    # 准确性基准测试
```

#### pytest 配置
```python
# conftest.py
import pytest
import torch
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def device():
    """GPU/CPU device fixture"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def sample_phase_data():
    """Generate sample phase data for testing"""
    # Create synthetic phase map
    y, x = np.ogrid[:256, :256]
    center = np.array([128, 128])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Spherical aberration
    phase = 0.1 * (r/128)**4

    return torch.from_numpy(phase).float()

@pytest.fixture
def mock_network():
    """Mock network for testing"""
    from src.fizeau_network import FizeauPhysicsNet
    return FizeauPhysicsNet()
```

### 单元测试

#### 物理模型测试
```python
# test_physics_models.py
import pytest
import torch
import numpy as np
from src.physics_models import AiryPSF, ZernikeBasis

class TestAiryPSF:
    def test_psf_normalization(self, device):
        """Test PSF normalization"""
        psf_model = AiryPSF(wavelength=632e-9, f_number=5.6)
        psf = psf_model.generate_psf(image_size=(256, 256))

        # Check normalization
        assert torch.allclose(torch.sum(psf), torch.tensor(1.0), atol=1e-6)

    def test_psf_symmetry(self, device):
        """Test PSF symmetry"""
        psf_model = AiryPSF(wavelength=632e-9, f_number=5.6)
        psf = psf_model.generate_psf(image_size=(256, 256))

        # Check center symmetry
        center = 128
        assert torch.allclose(psf[center, center-5:center+6],
                            torch.flip(psf[center, center-5:center+6], [0]))

class TestZernikeBasis:
    def test_orthogonality(self, sample_phase_data):
        """Test Zernike polynomial orthogonality"""
        zernike = ZernikeBasis(num_modes=37)

        # Project and reconstruct
        coeffs = zernike.project(sample_phase_data)
        reconstructed = zernike.reconstruct(coeffs)

        # Check reconstruction accuracy
        mse = torch.mean((reconstructed - sample_phase_data)**2)
        assert mse < 1e-10

    def test_zernike_modes(self):
        """Test individual Zernike modes"""
        zernike = ZernikeBasis(num_modes=10)

        for i in range(10):
            # Create coefficient vector with single mode
            coeffs = torch.zeros(10)
            coeffs[i] = 1.0

            # Reconstruct mode
            mode = zernike.reconstruct(coeffs)

            # Check non-zero values
            assert torch.sum(mode != 0) > 100
```

#### 网络组件测试
```python
# test_network_components.py
import pytest
import torch
from src.fizeau_network import DWTLayer, UnrollingBlock, ZernikeSupervisor

class TestDWTLayer:
    def test_dwt_forward(self, sample_phase_data):
        """Test DWT forward transform"""
        dwt = DWTLayer(wavelet='haar')
        coeffs = dwt.forward(sample_phase_data)

        # Check output structure
        assert isinstance(coeffs, dict)
        assert 'approx' in coeffs
        assert 'detail' in coeffs

    def test_dwt_inverse(self, sample_phase_data):
        """Test DWT inverse transform"""
        dwt = DWTLayer(wavelet='haar')

        # Forward and inverse
        coeffs = dwt.forward(sample_phase_data)
        reconstructed = dwt.inverse(coeffs)

        # Check reconstruction
        mse = torch.mean((reconstructed - sample_phase_data)**2)
        assert mse < 1e-6

class TestUnrollingBlock:
    def test_unrolling_forward(self, sample_phase_data):
        """Test unrolling block forward pass"""
        block = UnrollingBlock(physics_model=None)

        # Create mock interferogram
        interferogram = torch.abs(torch.exp(1j * sample_phase_data))**2

        output = block.forward(sample_phase_data, interferogram)

        # Check output shape
        assert output.shape == sample_phase_data.shape

    def test_gradient_flow(self, sample_phase_data):
        """Test gradient flow through unrolling block"""
        block = UnrollingBlock(physics_model=None)
        interferogram = torch.abs(torch.exp(1j * sample_phase_data))**2

        # Enable gradients
        sample_phase_data.requires_grad_(True)

        output = block.forward(sample_phase_data, interferogram)
        loss = torch.mean(output**2)
        loss.backward()

        # Check gradients
        assert sample_phase_data.grad is not None
        assert torch.sum(torch.abs(sample_phase_data.grad)) > 0
```

### 集成测试

#### 端到端测试
```python
# test_integration.py
import pytest
import torch
from src.fizeau_network import FizeauPhysicsNet
from src.config import Config

class TestEndToEnd:
    def test_full_pipeline(self, device):
        """Test complete phase retrieval pipeline"""
        # Load configuration
        config = Config()
        config.load('config/config.yaml')

        # Create network
        network = FizeauPhysicsNet(config.network)

        # Generate test data
        phase_gt = self.generate_test_phase()
        interferogram = self.generate_interferogram(phase_gt)

        # Run inference
        with torch.no_grad():
            phase_pred = network(interferogram.to(device))

        # Calculate metrics
        mse = torch.mean((phase_pred - phase_gt.to(device))**2)
        psnr = 20 * torch.log10(torch.max(phase_gt) / torch.sqrt(mse))

        # Assert reasonable performance
        assert psnr > 20.0  # At least 20dB PSNR

    def generate_test_phase(self):
        """Generate synthetic test phase"""
        # Create smooth phase map
        y, x = np.ogrid[:256, :256]
        phase = 0.5 * np.sin(2 * np.pi * x / 64) * np.sin(2 * np.pi * y / 64)
        return torch.from_numpy(phase).float()

    def generate_interferogram(self, phase):
        """Generate interferogram from phase"""
        field = torch.exp(1j * phase)
        intensity = torch.abs(field)**2
        return intensity / torch.max(intensity)
```

### 性能基准测试

#### 推理性能
```python
# benchmarks/performance_benchmarks.py
import time
import torch
import pytest
from src.fizeau_network import FizeauPhysicsNet

@pytest.mark.benchmark
def test_inference_speed(benchmark, device):
    """Benchmark inference speed"""
    network = FizeauPhysicsNet().to(device)
    network.eval()

    # Create test input
    test_input = torch.randn(1, 1, 512, 512).to(device)

    def run_inference():
        with torch.no_grad():
            _ = network(test_input)

    # Run benchmark
    benchmark(run_inference)

@pytest.mark.benchmark
def test_memory_usage(device):
    """Test memory usage during inference"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    network = FizeauPhysicsNet().to(device)

    # Clear cache
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # Run inference
    test_input = torch.randn(1, 1, 512, 512).to(device)
    with torch.no_grad():
        _ = network(test_input)

    final_memory = torch.cuda.memory_allocated()
    memory_used = final_memory - initial_memory

    # Assert reasonable memory usage (< 2GB)
    assert memory_used < 2 * 1024**3
```

#### 训练性能
```python
def test_training_speed(device):
    """Test training iteration speed"""
    network = FizeauPhysicsNet().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

    # Create batch data
    batch_size = 4
    input_data = torch.randn(batch_size, 1, 256, 256).to(device)
    target = torch.randn(batch_size, 1, 256, 256).to(device)

    start_time = time.time()

    # Training step
    optimizer.zero_grad()
    output = network(input_data)
    loss = torch.mean((output - target)**2)
    loss.backward()
    optimizer.step()

    end_time = time.time()

    iteration_time = end_time - start_time

    # Assert reasonable training speed (< 1 second per iteration)
    assert iteration_time < 1.0
```

### 验证测试

#### 准确性验证
```python
# benchmarks/accuracy_benchmarks.py
import numpy as np
from skimage.metrics import structural_similarity as ssim

def test_phase_accuracy():
    """Test phase retrieval accuracy on synthetic data"""
    # Generate known phase
    phase_gt = generate_known_phase()

    # Add noise
    noisy_interferogram = add_noise_to_interferogram(phase_gt, snr=20)

    # Retrieve phase
    phase_pred = retrieve_phase(noisy_interferogram)

    # Calculate metrics
    rmse = np.sqrt(np.mean((phase_pred - phase_gt)**2))
    psnr = 20 * np.log10(np.max(np.abs(phase_gt)) / rmse)
    ssim_val = ssim(phase_gt, phase_pred, data_range=phase_gt.max() - phase_gt.min())

    # Assert accuracy thresholds
    assert psnr > 25.0
    assert ssim_val > 0.8
    assert rmse < 0.1

def generate_known_phase():
    """Generate phase with known characteristics"""
    # Create phase with specific Zernike modes
    zernike = ZernikeBasis(num_modes=10)
    coeffs = torch.tensor([0, 0.1, 0.05, 0.02, 0, 0, 0, 0, 0, 0])
    return zernike.reconstruct(coeffs).numpy()
```

### 持续集成

#### GitHub Actions 配置
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10]

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        file: ./coverage.xml
```

#### 代码质量检查
```yaml
# .github/workflows/lint.yml
name: Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install linting tools
      run: |
        pip install flake8 black isort mypy

    - name: Run linters
      run: |
        flake8 src tests
        black --check src tests
        isort --check-only src tests
        mypy src
```

### 测试数据管理

#### 合成数据生成
```python
def generate_synthetic_dataset(num_samples=1000, image_size=(256, 256)):
    """
    Generate synthetic training/validation dataset

    Args:
        num_samples: Number of samples to generate
        image_size: Image dimensions

    Returns:
        dataset: Dictionary with phase and interferogram pairs
    """
    dataset = {'phase': [], 'interferogram': []}

    for i in range(num_samples):
        # Generate random Zernike coefficients
        coeffs = torch.randn(37) * 0.1

        # Create phase
        zernike = ZernikeBasis(37)
        phase = zernike.reconstruct(coeffs)

        # Generate interferogram
        interferogram = generate_interferogram_from_phase(phase)

        # Add noise
        noisy_interferogram = add_realistic_noise(interferogram)

        dataset['phase'].append(phase.numpy())
        dataset['interferogram'].append(noisy_interferogram.numpy())

    return dataset
```

#### 真实数据验证
```python
def validate_on_real_data(model, real_data_path):
    """
    Validate model on real measurement data

    Args:
        model: Trained model
        real_data_path: Path to real data

    Returns:
        metrics: Validation metrics
    """
    # Load real data
    real_data = load_real_measurements(real_data_path)

    metrics = {}

    for sample in real_data:
        # Run inference
        pred_phase = model(sample['interferogram'])

        # Calculate metrics against ground truth (if available)
        if 'gt_phase' in sample:
            mse = torch.mean((pred_phase - sample['gt_phase'])**2)
            metrics['mse'] = metrics.get('mse', []) + [mse.item()]

        # Qualitative metrics
        smoothness = calculate_smoothness(pred_phase)
        metrics['smoothness'] = metrics.get('smoothness', []) + [smoothness]

    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])

    return metrics
```

### 调试和诊断

#### 梯度检查
```python
def check_gradients(model, input_data, eps=1e-7):
    """Check gradient computation using finite differences"""
    model.eval()

    # Forward pass
    output = model(input_data)
    loss = torch.mean(output**2)

    # Analytical gradient
    loss.backward()
    analytical_grad = input_data.grad.clone()

    # Numerical gradient
    input_data.grad.zero_()
    numerical_grad = torch.zeros_like(input_data)

    for i in range(input_data.numel()):
        # Positive perturbation
        input_data_flat = input_data.view(-1)
        input_data_flat[i] += eps
        output_pos = model(input_data)
        loss_pos = torch.mean(output_pos**2)

        # Negative perturbation
        input_data_flat[i] -= 2 * eps
        output_neg = model(input_data)
        loss_neg = torch.mean(output_neg**2)

        # Central difference
        numerical_grad_flat = numerical_grad.view(-1)
        numerical_grad_flat[i] = (loss_pos - loss_neg) / (2 * eps)

        # Reset
        input_data_flat[i] += eps

    # Compare gradients
    diff = torch.abs(analytical_grad - numerical_grad)
    rel_error = diff / (torch.abs(analytical_grad) + 1e-8)

    return torch.max(rel_error).item()
```

#### 数值稳定性检查
```python
def check_numerical_stability(model, input_range=(-10, 10), num_samples=100):
    """Check model stability across input range"""
    model.eval()

    stability_issues = []

    for _ in range(num_samples):
        # Random input in range
        input_data = torch.rand(1, 1, 64, 64) * (input_range[1] - input_range[0]) + input_range[0]

        try:
            with torch.no_grad():
                output = model(input_data)

            # Check for NaN or Inf
            if torch.isnan(output).any() or torch.isinf(output).any():
                stability_issues.append("NaN/Inf detected")

            # Check output range
            if torch.max(torch.abs(output)) > 1000:
                stability_issues.append("Output magnitude too large")

        except Exception as e:
            stability_issues.append(f"Exception: {str(e)}")

    return stability_issues
```