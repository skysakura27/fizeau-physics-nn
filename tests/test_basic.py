"""
Basic functionality tests for Fizeau Physics NN.

This module contains tests to verify that the basic components
of the system are working correctly after installation.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

# Import project modules
from src.config import Config
from src.utils import create_circular_mask, normalize_phase
from tests import (
    TEST_DEVICE, TEST_DTYPE, TEST_IMAGE_SIZE,
    create_test_interferogram, create_test_phase,
    assert_tensor_shape, assert_no_nan_inf
)


class TestBasicFunctionality:
    """Test basic functionality of core components."""

    def test_config_loading(self):
        """Test that configuration can be loaded."""
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        config = Config(str(config_path))

        # Should not raise exception during initialization
        assert config.data is not None

        # Check that config has expected sections
        assert 'network' in config.data
        assert 'training' in config.data
        assert 'data' in config.data

    def test_utils_functions(self):
        """Test utility functions."""
        # Test circular mask creation
        mask = create_circular_mask(256, radius=0.45)  # Use single int for size
        assert_tensor_shape(mask, (256, 256))
        assert_no_nan_inf(mask)

        # Test phase normalization
        phase = create_test_phase()
        normalized = normalize_phase(phase)
        assert_tensor_shape(normalized, phase.shape)
        assert_no_nan_inf(normalized)

        # Check that normalization wraps phase to [-pi, pi]
        assert normalized.min() >= -np.pi
        assert normalized.max() <= np.pi
        assert normalized.min() >= -np.pi
        assert normalized.max() <= np.pi

    def test_tensor_operations(self):
        """Test basic tensor operations on target device."""
        # Create test tensors
        interferogram = create_test_interferogram()
        phase = create_test_phase()

        # Basic operations should work
        result = interferogram + phase
        assert_tensor_shape(result, interferogram.shape)
        assert_no_nan_inf(result)

        # Complex operations
        complex_field = torch.exp(1j * phase)
        intensity = torch.abs(complex_field)**2
        assert_tensor_shape(intensity, phase.shape)
        assert_no_nan_inf(intensity)

    def test_imports(self):
        """Test that main modules can be imported."""
        # Test core modules that should exist
        from src.config import Config
        from src.utils import create_circular_mask, normalize_phase

        # Test that fizeau_network can be imported (may be incomplete)
        try:
            from src.fizeau_network import FizeauPhysicsNet
        except ImportError:
            # Expected during development if not fully implemented
            pass

        # Physics models may not exist yet
        try:
            from src.physics_models import AiryPSF, ZernikeBasis
        except ImportError:
            # Expected during development
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_operations(self):
        """Test GPU operations if CUDA is available."""
        # Move tensors to GPU
        interferogram = create_test_interferogram(device='cpu')
        interferogram_gpu = interferogram.to('cuda')

        # Operations on GPU should work
        result = interferogram_gpu * 2
        assert result.device.type == 'cuda'
        assert_no_nan_inf(result)


class TestDataValidation:
    """Test data validation and preprocessing."""

    def test_interferogram_properties(self):
        """Test that generated interferograms have expected properties."""
        interferogram = create_test_interferogram()

        # Should be in [0, 1] range for normalized interferograms
        assert interferogram.min() >= 0
        assert interferogram.max() <= 1

    def test_phase_properties(self):
        """Test that generated phases have expected properties."""
        phase = create_test_phase()

        # Phase should be reasonably bounded
        assert phase.abs().max() < 10  # Less than 10 radians variation

    def test_mask_properties(self):
        """Test circular mask properties."""
        mask = create_circular_mask(TEST_IMAGE_SIZE, radius=50)

        # Mask should be binary (0 or 1)
        unique_values = torch.unique(mask)
        assert len(unique_values) <= 2  # Only 0 and 1

        # Should have some masked and unmasked pixels
        assert mask.sum() > 0  # Some pixels unmasked
        assert mask.sum() < mask.numel()  # Some pixels masked


class TestNumericalStability:
    """Test numerical stability of operations."""

    def test_gradient_computation(self):
        """Test that gradients can be computed."""
        phase = create_test_phase()
        phase.requires_grad_(True)

        # Simple operation
        result = torch.sum(phase**2)
        result.backward()

        assert phase.grad is not None
        assert_no_nan_inf(phase.grad)

    def test_large_tensor_operations(self):
        """Test operations on larger tensors."""
        large_size = (1024, 1024)
        large_tensor = torch.randn(1, 1, *large_size, device=TEST_DEVICE)

        # FFT operations (common in interferometry)
        fft_result = torch.fft.fft2(large_tensor)
        ifft_result = torch.fft.ifft2(fft_result)

        assert_tensor_shape(ifft_result, large_tensor.shape)
        # Allow small numerical errors
        diff = torch.abs(ifft_result - large_tensor)
        assert diff.max() < 1e-6


if __name__ == "__main__":
    # Run basic tests
    test_basic = TestBasicFunctionality()
    test_basic.test_config_loading()
    test_basic.test_utils_functions()
    test_basic.test_tensor_operations()

    print("Basic functionality tests passed!")

    test_data = TestDataValidation()
    test_data.test_interferogram_properties()
    test_data.test_phase_properties()
    test_data.test_mask_properties()

    print("Data validation tests passed!")

    test_stability = TestNumericalStability()
    test_stability.test_gradient_computation()

    print("Numerical stability tests passed!")
    print("All basic tests completed successfully!")