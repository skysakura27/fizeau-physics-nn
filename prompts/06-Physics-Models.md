# 06-Physics-Models.md

## Physics Models for Fizeau Interferometry

### 光学基础

#### Fizeau干涉仪原理
Fizeau干涉仪通过测量两束光的相位差来重建表面形貌：

```python
def fizeau_interferogram(phase, wavelength=632.8e-9):
    """
    Generate interferogram from phase map

    Args:
        phase: Phase map in radians
        wavelength: Light wavelength in meters

    Returns:
        interferogram: Intensity pattern (0-1 normalized)
    """
    # Convert phase to optical path difference
    opd = phase * wavelength / (2 * np.pi)

    # Interference pattern
    intensity = 0.5 * (1 + np.cos(2 * np.pi * opd / wavelength))

    return intensity
```

#### 相位展开问题
干涉图只提供包裹相位，需要展开：

```python
def phase_unwrapping(interferogram, method='quality-guided'):
    """
    Unwrap phase from interferogram

    Args:
        interferogram: Wrapped phase map
        method: Unwrapping algorithm

    Returns:
        unwrapped_phase: Continuous phase map
    """
    if method == 'quality-guided':
        # Quality-guided path following
        quality_map = compute_quality_map(interferogram)
        unwrapped = quality_guided_unwrap(interferogram, quality_map)
    elif method == 'least-squares':
        # Least squares integration
        unwrapped = least_squares_unwrap(interferogram)

    return unwrapped
```

### Airy衍射模型

#### 点扩散函数
```python
def airy_psf(wavelength, f_number, pixel_size, image_size):
    """
    Generate Airy disk PSF

    Args:
        wavelength: Light wavelength
        f_number: F-number of optical system
        pixel_size: Detector pixel size
        image_size: Image dimensions

    Returns:
        psf: Point spread function
    """
    # Airy disk radius
    airy_radius = 1.22 * wavelength * f_number

    # Convert to pixels
    airy_pixels = airy_radius / pixel_size

    # Generate coordinate grid
    y, x = np.ogrid[:image_size[0], :image_size[1]]
    center = np.array(image_size) / 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

    # Normalized radius
    rho = r / airy_pixels

    # Airy pattern
    psf = (2 * j1(np.pi * rho) / (np.pi * rho))**2

    # Normalize
    psf = psf / np.sum(psf)

    return psf
```

#### 光学传递函数
```python
class OpticalTransferFunction(nn.Module):
    def __init__(self, wavelength, f_number, pixel_size, image_size):
        super().__init__()
        self.psf = self.compute_otf(wavelength, f_number, pixel_size, image_size)

    def compute_otf(self, wavelength, f_number, pixel_size, image_size):
        """Compute optical transfer function"""
        psf = airy_psf(wavelength, f_number, pixel_size, image_size)

        # FFT to get OTF
        otf = np.fft.fft2(psf)
        otf = np.fft.fftshift(otf)

        return torch.from_numpy(otf).float()

    def forward(self, phase):
        """Apply optical system to phase"""
        # Convert phase to complex field
        field = torch.exp(1j * phase)

        # Apply OTF in frequency domain
        field_fft = torch.fft.fft2(field)
        field_fft = torch.fft.fftshift(field_fft)

        filtered_fft = field_fft * self.psf.to(field.device)

        # Inverse FFT
        filtered_field = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft))

        return torch.angle(filtered_field)
```

### 相位重建算法

#### 传统方法
```python
def traditional_phase_retrieval(interferogram, iterations=10):
    """
    Traditional phase retrieval using Gerchberg-Saxton algorithm

    Args:
        interferogram: Measured intensity
        iterations: Number of iterations

    Returns:
        phase: Retrieved phase
    """
    # Initial random phase
    phase = torch.randn_like(interferogram) * 2 * np.pi

    for i in range(iterations):
        # Forward propagation
        field = torch.exp(1j * phase)
        intensity = torch.abs(field)**2

        # Replace magnitude with measured intensity
        new_field = torch.sqrt(interferogram) * torch.exp(1j * torch.angle(field))

        # Backward propagation (simplified)
        phase = torch.angle(new_field)

    return phase
```

#### 物理信息神经网络
```python
class PhysicsInformedPhaseRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.optics_model = OpticalTransferFunction(...)
        self.neural_net = UnrolledNetwork()

    def forward(self, interferogram):
        # Initial estimate
        phase = self.neural_net.init_phase(interferogram)

        # Unrolled iterations with physics
        for block in self.neural_net.blocks:
            # Neural update
            phase_neural = block(phase, interferogram)

            # Physics constraint
            phase_phys = self.optics_model(phase_neural)

            # Combine
            phase = 0.7 * phase_neural + 0.3 * phase_phys

        return phase

    def physics_loss(self, pred_phase, interferogram):
        """Physics-informed loss"""
        # Forward model
        pred_intensity = self.optics_model.forward_intensity(pred_phase)

        # Data fidelity
        data_loss = F.mse_loss(pred_intensity, interferogram)

        # Smoothness regularization
        smoothness_loss = torch.mean(torch.abs(pred_phase[:, 1:] - pred_phase[:, :-1]))

        return data_loss + 0.01 * smoothness_loss
```

### 噪声模型

#### 系统噪声
```python
def system_noise_model(image_size, noise_params):
    """
    Generate system noise pattern

    Args:
        image_size: Image dimensions
        noise_params: Dictionary of noise parameters

    Returns:
        noise_pattern: System noise
    """
    # Fixed pattern noise
    fpn = torch.randn(image_size) * noise_params.get('fpn_std', 0.01)

    # Dark current
    dark_current = torch.ones(image_size) * noise_params.get('dark_current', 0.001)

    # Shot noise (Poisson)
    signal_level = noise_params.get('signal_level', 1000)
    shot_noise = torch.poisson(torch.ones(image_size) * signal_level) - signal_level
    shot_noise = shot_noise / signal_level

    return fpn + dark_current + shot_noise
```

#### 相位噪声
```python
def phase_noise_model(clean_phase, snr_db=30):
    """
    Add phase noise to clean phase

    Args:
        clean_phase: Clean phase map
        snr_db: Signal-to-noise ratio

    Returns:
        noisy_phase: Phase with noise
    """
    # Phase noise standard deviation
    phase_noise_std = np.sqrt(2) / (10**(snr_db/20))

    # Add Gaussian noise
    noise = torch.randn_like(clean_phase) * phase_noise_std

    return clean_phase + noise
```

### 校正算法

#### 像差校正
```python
class AberrationCorrection(nn.Module):
    def __init__(self, num_zernike_modes=37):
        super().__init__()
        self.zernike_basis = ZernikeBasis(num_zernike_modes)

    def forward(self, phase):
        # Project to Zernike space
        coeffs = self.zernike_basis.project(phase)

        # Remove low-order aberrations
        coeffs[:10] = 0  # Remove piston, tilt, defocus, etc.

        # Reconstruct corrected phase
        corrected_phase = self.zernike_basis.reconstruct(coeffs)

        return phase - corrected_phase
```

#### 非线性效应校正
```python
def nonlinear_correction(phase, intensity, max_intensity=1.0):
    """
    Correct for detector nonlinearity

    Args:
        phase: Raw phase
        intensity: Measured intensity
        max_intensity: Saturation level

    Returns:
        corrected_phase: Nonlinearity-corrected phase
    """
    # Gamma correction
    gamma = 2.2
    corrected_intensity = intensity**(1/gamma)

    # Rescale
    corrected_intensity = corrected_intensity * max_intensity

    # Update phase estimate
    corrected_phase = torch.angle(torch.sqrt(corrected_intensity) * torch.exp(1j * phase))

    return corrected_phase
```

### 性能评估

#### 准确性指标
- **相位误差**: RMS 相位误差
- **表面精度**: PV 和 RMS 表面误差
- **重复性**: 多次测量的标准差

#### 鲁棒性测试
- **信噪比范围**: 10dB 到 40dB
- **像差范围**: 不同 Zernike 系数
- **环境条件**: 温度、振动影响

### 实际应用优化

- **实时处理**: GPU 加速计算
- **内存优化**: 分块处理大图像
- **自适应参数**: 根据测量条件调整