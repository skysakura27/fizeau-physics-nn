"""Wavelet operations for frequency domain decoupling."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTPreprocessor(nn.Module):
    """
    2D Discrete Wavelet Transform preprocessor for frequency domain decoupling.

    Implements frequency domain decoupling by separating signal and noise components
    using wavelet decomposition.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.wavelet = config.get("wavelet", "haar")
        self.levels = config.get("levels", 2)

        # Haar wavelet filters (simplified implementation)
        # In practice, you might want to use PyWavelets or implement full DWT
        self.haar_filters = self._get_haar_filters()

    def _get_haar_filters(self) -> Dict[str, torch.Tensor]:
        """Get Haar wavelet decomposition filters."""
        # Low-pass filter
        low = torch.tensor([0.7071, 0.7071]).view(1, 1, 1, 2)
        # High-pass filter
        high = torch.tensor([-0.7071, 0.7071]).view(1, 1, 1, 2)

        return {"low": low, "high": high}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply 2D DWT decomposition.

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            Dictionary with decomposition coefficients
        """
        # Simplified 1-level DWT for demonstration
        # In practice, implement full multi-level DWT

        # Low-low (approximation coefficients)
        ll = F.conv2d(x, self.haar_filters["low"], stride=2)
        ll = F.conv2d(ll, self.haar_filters["low"].transpose(-1, -2), stride=2)

        # Low-high
        lh = F.conv2d(x, self.haar_filters["low"], stride=2)
        lh = F.conv2d(lh, self.haar_filters["high"].transpose(-1, -2), stride=2)

        # High-low
        hl = F.conv2d(x, self.haar_filters["high"], stride=2)
        hl = F.conv2d(hl, self.haar_filters["low"].transpose(-1, -2), stride=2)

        # High-high (detail coefficients - contains speckle noise)
        hh = F.conv2d(x, self.haar_filters["high"], stride=2)
        hh = F.conv2d(hh, self.haar_filters["high"].transpose(-1, -2), stride=2)

        return {
            "LL": ll,  # Low-frequency approximation (useful signal)
            "LH": lh,  # Horizontal details
            "HL": hl,  # Vertical details
            "HH": hh,  # Diagonal details (speckle noise)
        }


class DWTPreprocessLayer(nn.Module):
    """
    Fixed 2D Haar DWT preprocessing layer.

    Takes an intensity image (B, 1, H, W) and returns concatenated
    wavelet sub-bands (B, 4, H/2, W/2) in the order: LL, LH, HL, HH.
    """

    def __init__(self):
        super().__init__()
        weight = self._build_haar_kernels()
        self.register_buffer("haar_weight", weight)

    @staticmethod
    def _build_haar_kernels() -> torch.Tensor:
        """Create 2x2 Haar kernels for LL, LH, HL, HH."""
        low = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
        high = torch.tensor([-1.0, 1.0]) / math.sqrt(2.0)

        ll = torch.ger(low, low)
        lh = torch.ger(low, high)
        hl = torch.ger(high, low)
        hh = torch.ger(high, high)

        weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Intensity image tensor (B, 1, H, W)

        Returns:
            Concatenated sub-bands (B, 4, H/2, W/2)
        """
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError("DWTPreprocessLayer expects input shape (B, 1, H, W).")
        if x.size(-1) % 2 != 0 or x.size(-2) % 2 != 0:
            raise ValueError("Input H and W must be even for 2x2 Haar DWT.")

        weight = self.haar_weight.to(dtype=x.dtype, device=x.device)
        return F.conv2d(x, weight, stride=2)
