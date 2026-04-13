"""Common utility functions."""

import numpy as np
import torch


def create_circular_mask(size: int, radius: float = 0.45) -> torch.Tensor:
    """
    Create a circular pupil mask.

    Args:
        size: Image size (height = width)
        radius: Normalized (0-1)

    Returns:
        Circular mask tensor of shape (size, size)
    """
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    mask = (dist <= radius * center).astype(np.float32)
    return torch.from_numpy(mask)


def normalize_phase(phase: torch.Tensor) -> torch.Tensor:
    """Normalize phase to [-π, π] range."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def compute_rms_error(phase: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute RMS wavefront error."""
    phase_masked = phase[mask > 0]
    phase_centered = phase_masked - phase_masked.mean()
    return float(torch.sqrt(torch.mean(phase_centered ** 2)))
