"""Utility functions for the Fizeau PINN."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List
import math


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
    dist = np.sqrt((x - center)**2 + (y - center)**2)
    mask = (dist <= radius * center).astype(np.float32)
    return torch.from_numpy(mask)


def normalize_phase(phase: torch.Tensor) -> torch.Tensor:
    """Normalize phase to [-π, π] range."""
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def compute_rms_error(phase: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute RMS wavefront error."""
    phase_masked = phase[mask > 0]
    phase_centered = phase_masked - phase_masked.mean()
    return float(torch.sqrt(torch.mean(phase_centered**2)))


class DWTPreprocessor(nn.Module):
    """
    2D Discrete Wavelet Transform preprocessor for frequency domain decoupling.

    Implements frequency domain decoupling by separating signal and noise components
    using wavelet decomposition.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.wavelet = config.get('wavelet', 'haar')
        self.levels = config.get('levels', 2)

        # Haar wavelet filters (simplified implementation)
        # In practice, you might want to use PyWavelets or implement full DWT
        self.haar_filters = self._get_haar_filters()

    def _get_haar_filters(self):
        """Get Haar wavelet decomposition filters."""
        # Low-pass filter
        low = torch.tensor([0.7071, 0.7071]).view(1, 1, 1, 2)
        # High-pass filter
        high = torch.tensor([-0.7071, 0.7071]).view(1, 1, 1, 2)

        return {'low': low, 'high': high}

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
        ll = F.conv2d(x, self.haar_filters['low'], stride=2)
        ll = F.conv2d(ll, self.haar_filters['low'].transpose(-1, -2), stride=2)

        # Low-high
        lh = F.conv2d(x, self.haar_filters['low'], stride=2)
        lh = F.conv2d(lh, self.haar_filters['high'].transpose(-1, -2), stride=2)

        # High-low
        hl = F.conv2d(x, self.haar_filters['high'], stride=2)
        hl = F.conv2d(hl, self.haar_filters['low'].transpose(-1, -2), stride=2)

        # High-high (detail coefficients - contains speckle noise)
        hh = F.conv2d(x, self.haar_filters['high'], stride=2)
        hh = F.conv2d(hh, self.haar_filters['high'].transpose(-1, -2), stride=2)

        return {
            'LL': ll,  # Low-frequency approximation (useful signal)
            'LH': lh,  # Horizontal details
            'HL': hl,  # Vertical details
            'HH': hh   # Diagonal details (speckle noise)
        }


class ZernikeBasis:
    """
    Zernike polynomial basis for wavefront representation.

    Implements 36-order Zernike polynomials for hard constraint output.
    """

    def __init__(self, num_modes: int = 36, image_size: int = 512):
        self.num_modes = num_modes
        self.image_size = image_size

        # Pre-compute Zernike basis matrices
        self.basis_matrices = self._compute_zernike_basis()

    def _compute_zernike_basis(self) -> torch.Tensor:
        """Compute Zernike polynomial basis matrices."""
        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size),
            torch.linspace(-1, 1, self.image_size),
            indexing='ij'
        )

        # Convert to polar coordinates
        rho = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)

        # Mask for unit circle
        mask = rho <= 1.0

        # Initialize basis matrices
        basis = torch.zeros(self.num_modes, self.image_size, self.image_size)

        # Zernike polynomials up to order 6 (36 modes)
        zernike_coeffs = self._get_zernike_coefficients()

        for i in range(min(self.num_modes, len(zernike_coeffs))):
            n, m = zernike_coeffs[i]
            basis[i] = self._zernike_polynomial(n, m, rho, theta) * mask

        return basis

    def _get_zernike_coefficients(self) -> List[Tuple[int, int]]:
        """Get (n, m) coefficients for Zernike polynomials."""
        coeffs = []
        for n in range(7):  # Up to 6th order
            for m in range(-n, n+1, 2):
                coeffs.append((n, m))
        return coeffs[:self.num_modes]

    def _zernike_polynomial(self, n: int, m: int, rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Compute individual Zernike polynomial."""
        # Radial polynomial R_n^|m|(ρ)
        R = self._radial_polynomial(n, abs(m), rho)

        # Angular part
        if m >= 0:
            Z = R * torch.cos(m * theta)
        else:
            Z = R * torch.sin(abs(m) * theta)

        return Z

    def _radial_polynomial(self, n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
        """Compute radial polynomial R_n^m(ρ)."""
        if n == 0 and m == 0:
            return torch.ones_like(rho)

        R = torch.zeros_like(rho)
        for k in range((n - m) // 2 + 1):
            coeff = ((-1)**k * math.factorial(n - k) /
                    (math.factorial(k) * math.factorial((n + m) // 2 - k) *
                     math.factorial((n - m) // 2 - k)))
            R += coeff * rho**(n - 2*k)

        return R

    def project(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Project phase onto Zernike basis.

        Args:
            phase: Phase map (B, 1, H, W) or (H, W)

        Returns:
            coefficients: Zernike coefficients (B, num_modes) or (num_modes,)
        """
        if phase.dim() == 4:  # Batch dimension
            # Compute dot product with basis
            coeffs = torch.einsum('bchw,mhw->bm', phase.squeeze(1), self.basis_matrices)
            # Normalize by basis norm
            basis_norms = torch.einsum('mhw,mhw->m', self.basis_matrices, self.basis_matrices)
            coeffs = coeffs / basis_norms.unsqueeze(0)
        else:  # Single image
            coeffs = torch.einsum('hw,mhw->m', phase, self.basis_matrices)
            basis_norms = torch.einsum('mhw,mhw->m', self.basis_matrices, self.basis_matrices)
            coeffs = coeffs / basis_norms

        return coeffs

    def reconstruct(self, coefficients: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct phase from Zernike coefficients.

        Args:
            coefficients: Zernike coefficients (B, num_modes) or (num_modes,)

        Returns:
            phase: Reconstructed phase (B, 1, H, W) or (H, W)
        """
        if coefficients.dim() == 2:  # Batch dimension
            phase = torch.einsum('bm,mhw->bhw', coefficients, self.basis_matrices)
            phase = phase.unsqueeze(1)  # Add channel dimension
        else:  # Single set of coefficients
            phase = torch.einsum('m,mhw->hw', coefficients, self.basis_matrices)

        return phase


class AiryPhysicsModel:
    """
    Airy diffraction physics model for interferometry.

    Implements the physical forward model and gradient computation
    for multi-beam interference.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.wavelength = config.get('wavelength', 632.8e-9)  # 632.8nm
        self.finesse = config.get('finesse', 10)  # Cavity finesse

    def forward_model(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Physical forward model: phase -> interferogram.

        Implements Airy function for multi-beam interference:
        I = I_max / (1 + F * sin²(φ/2))

        Args:
            phase: Phase map (B, 1, H, W)

        Returns:
            interferogram: Predicted interferogram (B, 1, H, W)
        """
        # Simplified Airy function
        # In practice, this would be more complex
        phi = phase.squeeze(1)
        I_max = 1.0
        F = self.finesse

        intensity = I_max / (1 + F * torch.sin(phi / 2)**2)

        return intensity.unsqueeze(1)

    def compute_gradient(self, phase: torch.Tensor, measured_intensity: torch.Tensor) -> torch.Tensor:
        """
        Compute Airy physics gradient for phase correction.

        ∇_φℒ_Airy = dI_pred/dφ * (I_meas - I_pred)

        Args:
            phase: Current phase estimate (B, 1, H, W)
            measured_intensity: Measured interferogram (B, 1, H, W)

        Returns:
            gradient: Phase gradient (B, 1, H, W)
        """
        phi = phase.squeeze(1)
        I_meas = measured_intensity.squeeze(1)

        # Resize measured intensity to phase resolution if needed
        if I_meas.shape != phi.shape:
            I_meas = F.adaptive_avg_pool2d(I_meas.unsqueeze(1), phi.shape[-2:]).squeeze(1)

        # Derivative of Airy function w.r.t. phase
        finesse = self.finesse
        dI_dphi = -finesse * torch.sin(phi) / (2 * (1 + finesse * torch.sin(phi / 2)**2)**2)

        # Predicted intensity
        I_pred = self.forward_model(phase).squeeze(1)

        # Gradient
        gradient = dI_dphi * (I_meas - I_pred)

        return gradient.unsqueeze(1)
