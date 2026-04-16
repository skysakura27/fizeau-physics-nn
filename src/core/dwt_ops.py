"""Wavelet operations for frequency domain decoupling."""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_haar_weight(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Create 2x2 Haar kernels for LL, LH, HL, HH."""
    low = torch.tensor([1.0, 1.0], dtype=dtype, device=device) / math.sqrt(2.0)
    high = torch.tensor([-1.0, 1.0], dtype=dtype, device=device) / math.sqrt(2.0)

    ll = torch.ger(low, low)
    lh = torch.ger(low, high)
    hl = torch.ger(high, low)
    hh = torch.ger(high, high)

    weight = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
    return weight


def _pad_to_even(x: torch.Tensor, pad_mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Pad to even H/W by adding at most one pixel on right/bottom."""
    pad_right = int(x.size(-1) % 2 != 0)
    pad_bottom = int(x.size(-2) % 2 != 0)
    pad = (0, pad_right, 0, pad_bottom)
    if pad_right or pad_bottom:
        x = F.pad(x, pad, mode=pad_mode)
    return x, pad


def _unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """Remove padding added by _pad_to_even."""
    _, pad_right, _, pad_bottom = pad
    if pad_bottom:
        x = x[..., :-pad_bottom, :]
    if pad_right:
        x = x[..., :, :-pad_right]
    return x


def wavelet_decomposition(
    x: torch.Tensor,
    levels: int = 1,
    wavelet: str = "haar",
    pad_mode: str = "reflect",
) -> List[Dict[str, torch.Tensor]]:
    """
    Perform multi-level 2D wavelet decomposition (Haar fallback).

    Args:
        x: Input tensor (B, 1, H, W)
        levels: Decomposition levels
        wavelet: Wavelet basis (only 'haar' supported in fallback)
        pad_mode: Padding mode for odd spatial sizes

    Returns:
        coeffs: List of dicts with LL/LH/HL/HH and pad info per level
    """
    if wavelet != "haar":
        raise NotImplementedError("Only 'haar' wavelet is supported in the fallback implementation.")
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError("Input must have shape (B, 1, H, W).")
    if levels < 1:
        raise ValueError("levels must be >= 1.")

    coeffs: List[Dict[str, torch.Tensor]] = []
    current = x
    for _ in range(levels):
        current, pad = _pad_to_even(current, pad_mode=pad_mode)
        weight = _build_haar_weight(dtype=current.dtype, device=current.device)
        bands = F.conv2d(current, weight, stride=2)
        ll, lh, hl, hh = torch.chunk(bands, 4, dim=1)
        coeffs.append({"LL": ll, "LH": lh, "HL": hl, "HH": hh, "pad": pad})
        current = ll

    return coeffs


def wavelet_reconstruction(
    coeffs: List[Dict[str, torch.Tensor]],
    wavelet: str = "haar",
) -> torch.Tensor:
    """
    Reconstruct signal from wavelet coefficients (Haar fallback).

    Args:
        coeffs: List of dicts with LL/LH/HL/HH and pad info per level
        wavelet: Wavelet basis (only 'haar' supported in fallback)

    Returns:
        reconstructed: Reconstructed signal (B, 1, H, W)
    """
    if wavelet != "haar":
        raise NotImplementedError("Only 'haar' wavelet is supported in the fallback implementation.")
    if not coeffs:
        raise ValueError("coeffs must be a non-empty list.")

    current = None
    for level in reversed(coeffs):
        ll = level["LL"] if current is None else current
        lh = level["LH"]
        hl = level["HL"]
        hh = level["HH"]

        bands = torch.cat([ll, lh, hl, hh], dim=1)
        weight = _build_haar_weight(dtype=bands.dtype, device=bands.device)
        current = F.conv_transpose2d(bands, weight, stride=2)
        current = _unpad(current, level.get("pad", (0, 0, 0, 0)))

    return current


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
        self.levels = config.get("levels", 1)
        self.pad_mode = config.get("pad_mode", "reflect")

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply 2D DWT decomposition.

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            Dictionary with decomposition coefficients
        """
        coeffs = wavelet_decomposition(
            x,
            levels=self.levels,
            wavelet=self.wavelet,
            pad_mode=self.pad_mode,
        )
        first = coeffs[0]
        return {
            "LL": first["LL"],
            "LH": first["LH"],
            "HL": first["HL"],
            "HH": first["HH"],
            "coeffs": coeffs,
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
        return _build_haar_weight(dtype=torch.float32, device=torch.device("cpu"))

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
