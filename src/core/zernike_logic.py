"""Zernike basis generation and coefficient projection utilities."""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


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


# ---------------------------------------------------------------------------
# ZernikeSupervisor — nn.Module with precomputed basis & pseudo-inverse
# ---------------------------------------------------------------------------

def _noll_to_nm(j: int) -> Tuple[int, int]:
    """Convert Noll index j (≥1) to Zernike radial order n and azimuthal frequency m."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    m_options = list(range(-n, n + 1, 2))
    k = j - n * (n + 1) // 2 - 1
    # Noll ordering: even j → positive m first, odd j → negative m
    m_sorted: list[int] = []
    neg = [v for v in m_options if v < 0]
    pos = [v for v in m_options if v > 0]
    zero = [v for v in m_options if v == 0]
    # interleave: |m| ascending, even j gets cos (m≥0), odd j gets sin (m<0)
    pairs: list[int] = []
    for a in range(len(pos)):
        pairs.append(pos[a])
        if a < len(neg):
            pairs.append(neg[len(neg) - 1 - a])
    m_sorted = zero + pairs
    m = m_sorted[min(k, len(m_sorted) - 1)]
    return n, m


def _radial_poly(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    """Radial Zernike polynomial R_n^|m|(rho)."""
    m_abs = abs(m)
    R = torch.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        coeff = (
            (-1) ** k
            * math.factorial(n - k)
            / (
                math.factorial(k)
                * math.factorial((n + m_abs) // 2 - k)
                * math.factorial((n - m_abs) // 2 - k)
            )
        )
        R = R + coeff * rho ** (n - 2 * k)
    return R


def build_zernike_basis(
    n_modes: int = 36,
    height: int = 128,
    width: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build Zernike basis matrix G and its pseudo-inverse.

    Returns:
        G:      (n_modes, H*W)
        G_pinv: (H*W, n_modes)
    """
    lin_y = torch.linspace(-1, 1, height)
    lin_x = torch.linspace(-1, 1, width)
    yy, xx = torch.meshgrid(lin_y, lin_x, indexing='ij')
    rho   = torch.sqrt(xx ** 2 + yy ** 2)
    theta = torch.atan2(yy, xx)
    mask  = (rho <= 1.0).float()

    G = torch.zeros(n_modes, height * width)
    for j in range(1, n_modes + 1):
        n, m = _noll_to_nm(j)
        R = _radial_poly(n, m, rho)
        if m >= 0:
            Z = R * torch.cos(m * theta)
        else:
            Z = R * torch.sin(abs(m) * theta)
        Z = Z * mask
        # Normalise each mode to unit energy inside the pupil
        norm = torch.sqrt((Z ** 2).sum() + 1e-12)
        Z = Z / norm
        G[j - 1] = Z.reshape(-1)

    G_pinv = torch.linalg.pinv(G)   # (H*W, n_modes)
    return G, G_pinv


class ZernikeSupervisor(nn.Module):
    """Hard Zernike projection layer for phase-map regularisation.

    Precomputes the first 36 Zernike polynomials (Noll index) on a
    128×128 circular pupil mask. In the forward pass:
        a = pinv(G) @ phi_flat          # solve for coefficients
        phi_rec = G @ a                 # reconstruct smooth phase

    This forces the output to be a valid optical wavefront spanned by
    the first 36 Zernike modes, removing grid-like noise.

    The basis is stored as non-trainable buffers; they follow .to(device)
    automatically and add zero parameters to the optimiser.
    """

    N_MODES: int = 36
    HEIGHT:  int = 128
    WIDTH:   int = 128

    def __init__(self, n_modes: int = 36, height: int = 128, width: int = 128):
        super().__init__()
        self.N_MODES = n_modes
        self.HEIGHT  = height
        self.WIDTH   = width

        G, G_pinv = build_zernike_basis(n_modes, height, width)
        self.register_buffer("G",      G)       # (n_modes, H*W)
        self.register_buffer("G_pinv", G_pinv)   # (H*W, n_modes)

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase: (B, 1, H, W) phase map in radians.
        Returns:
            (B, 1, H, W) Zernike-reconstructed phase map.
        """
        B, C, H, W = phase.shape
        if (H, W) != (self.HEIGHT, self.WIDTH):
            raise ValueError(
                f"ZernikeSupervisor expects ({self.HEIGHT}, {self.WIDTH}) "
                f"spatial dims, got ({H}, {W})."
            )

        flat   = phase.view(B * C, H * W)                       # (B, HW)
        coeffs = flat @ self.G_pinv                              # (B, n_modes)
        recon  = coeffs @ self.G                                 # (B, HW)
        return recon.view(B, C, H, W)
