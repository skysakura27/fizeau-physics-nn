"""Zernike basis generation and coefficient projection utilities."""

import math
from typing import List, Tuple

import torch


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
            indexing="ij",
        )

        # Convert to polar coordinates
        rho = torch.sqrt(x ** 2 + y ** 2)
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
            for m in range(-n, n + 1, 2):
                coeffs.append((n, m))
        return coeffs[: self.num_modes]

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
            coeff = (
                (-1) ** k
                * math.factorial(n - k)
                / (
                    math.factorial(k)
                    * math.factorial((n + m) // 2 - k)
                    * math.factorial((n - m) // 2 - k)
                )
            )
            R += coeff * rho ** (n - 2 * k)

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
            coeffs = torch.einsum("bchw,mhw->bm", phase.squeeze(1), self.basis_matrices)
            # Normalize by basis norm
            basis_norms = torch.einsum("mhw,mhw->m", self.basis_matrices, self.basis_matrices)
            coeffs = coeffs / basis_norms.unsqueeze(0)
        else:  # Single image
            coeffs = torch.einsum("hw,mhw->m", phase, self.basis_matrices)
            basis_norms = torch.einsum("mhw,mhw->m", self.basis_matrices, self.basis_matrices)
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
            phase = torch.einsum("bm,mhw->bhw", coefficients, self.basis_matrices)
            phase = phase.unsqueeze(1)  # Add channel dimension
        else:  # Single set of coefficients
            phase = torch.einsum("m,mhw->hw", coefficients, self.basis_matrices)

        return phase
