# tests/__init__.py

"""
Test package for Fizeau Physics NN.

This package contains comprehensive tests for all components of the
physics-informed neural network for Fizeau interferometry.
"""

# Test configuration
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Common test utilities
import torch
import numpy as np
from pathlib import Path

# Test constants
TEST_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DTYPE = torch.float32
TEST_IMAGE_SIZE = (256, 256)
TEST_BATCH_SIZE = 2

def create_test_interferogram(shape=None, device=None):
    """Create a synthetic interferogram for testing."""
    if shape is None:
        shape = TEST_IMAGE_SIZE
    if device is None:
        device = TEST_DEVICE

    # Create simple fringe pattern
    y, x = np.ogrid[:shape[0], :shape[1]]
    interferogram = 0.5 * (1 + np.cos(2 * np.pi * (x + y) / 32))

    return torch.from_numpy(interferogram).float().unsqueeze(0).unsqueeze(0).to(device)

def create_test_phase(shape=None, device=None):
    """Create a synthetic phase map for testing."""
    if shape is None:
        shape = TEST_IMAGE_SIZE
    if device is None:
        device = TEST_DEVICE

    # Create smooth phase variation
    y, x = np.ogrid[:shape[0], :shape[1]]
    phase = 0.1 * np.sin(2 * np.pi * x / 64) * np.sin(2 * np.pi * y / 64)

    return torch.from_numpy(phase).float().unsqueeze(0).unsqueeze(0).to(device)

def assert_tensor_shape(tensor, expected_shape):
    """Assert that tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

def assert_tensor_range(tensor, min_val=None, max_val=None):
    """Assert that tensor values are in expected range."""
    if min_val is not None:
        assert tensor.min() >= min_val, f"Tensor min {tensor.min()} < {min_val}"
    if max_val is not None:
        assert tensor.max() <= max_val, f"Tensor max {tensor.max()} > {max_val}"

def assert_no_nan_inf(tensor):
    """Assert that tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), "Tensor contains NaN values"
    assert not torch.isinf(tensor).any(), "Tensor contains Inf values"