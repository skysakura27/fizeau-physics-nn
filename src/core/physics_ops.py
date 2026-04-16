"""Physics-informed operations for Fizeau interferometry PINN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWTLayer(nn.Module):
    """2D Haar Discrete Wavelet Transform (single-level decomposition).

    Decomposes a 1-channel intensity image into 4 sub-band channels:
        LL (approximation), LH (horizontal detail),
        HL (vertical detail), HH (diagonal detail).

    Input:  (B, 1, H, W)
    Output: (B, 4, H//2, W//2)
    """

    def __init__(self):
        super().__init__()
        # Haar filters — fixed, non-trainable
        ll = torch.tensor([[ 1,  1], [ 1,  1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[ 1,  1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[ 1, -1], [ 1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[ 1, -1], [-1,  1]], dtype=torch.float32) * 0.5

        # Shape: (4, 1, 2, 2)
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("kernel", kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) — single-channel intensity image.
        Returns:
            (B, 4, H//2, W//2) — LL, LH, HL, HH sub-bands.
        """
        if x.shape[1] != 1:
            raise ValueError(f"DWTLayer expects 1-channel input, got {x.shape[1]}.")
        return F.conv2d(x, self.kernel, stride=2, padding=0)


class DWTDisentangler(nn.Module):
    """Normalised Haar DWT with a learnable artifact-free reconstructor.

    Forward (DWT):
        1. Min-max normalise input intensity to [0, 1] for gradient stability.
        2. Fixed 2D Haar wavelet decomposition.
        Input:  (B, 1, 128, 128)
        Output: (B, 4, 64, 64)  — LL, LH, HL, HH + saved norm params.

    Inverse (reconstruct):
        Bilinear upsample (×2) + 3×3 Conv fusion → (B, 1, 128, 128).
        Avoids the checkerboard artifacts inherent in ConvTranspose2d.
    """

    def __init__(self):
        super().__init__()
        # ---- Fixed Haar DWT kernels (same as DWTLayer) ----
        ll = torch.tensor([[ 1,  1], [ 1,  1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[ 1,  1], [-1, -1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[ 1, -1], [ 1, -1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[ 1, -1], [-1,  1]], dtype=torch.float32) * 0.5
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4,1,2,2)
        self.register_buffer("kernel", kernel)

        # ---- Learnable reconstructor: upsample + conv (no checkerboard) ----
        self.reconstructor = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
        )

    # ----- normalisation helpers -----
    @staticmethod
    def _normalize(x: torch.Tensor):
        """Per-sample min-max normalisation to [0, 1]."""
        B = x.shape[0]
        flat = x.view(B, -1)
        x_min = flat.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        x_max = flat.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
        scale = (x_max - x_min).clamp(min=1e-8)
        return (x - x_min) / scale, x_min, scale

    @staticmethod
    def _denormalize(x: torch.Tensor, x_min: torch.Tensor, scale: torch.Tensor):
        """Invert per-sample min-max normalisation."""
        return x * scale + x_min

    # ----- forward: DWT -----
    def forward(self, x: torch.Tensor):
        """Normalise and decompose.

        Args:
            x: (B, 1, H, W) intensity image.
        Returns:
            sub_bands: (B, 4, H//2, W//2)
            norm_params: (x_min, scale) needed by reconstruct() to undo normalisation.
        """
        if x.shape[1] != 1:
            raise ValueError(f"DWTDisentangler expects 1-channel input, got {x.shape[1]}.")
        x_norm, x_min, scale = self._normalize(x)
        sub_bands = F.conv2d(x_norm, self.kernel, stride=2, padding=0)
        return sub_bands, (x_min, scale)

    # ----- inverse: learnable IDWT -----
    def reconstruct(self, sub_bands: torch.Tensor, norm_params=None):
        """Reconstruct from sub-bands back to full resolution.

        Args:
            sub_bands:   (B, 4, H//2, W//2)
            norm_params: Optional (x_min, scale) from forward(); if given the
                         output is mapped back to the original intensity range.
        Returns:
            (B, 1, H, W)
        """
        out = self.reconstructor(sub_bands)       # (B, 1, H, W)
        if norm_params is not None:
            x_min, scale = norm_params
            out = self._denormalize(out, x_min, scale)
        return out


class AirySimulator(nn.Module):
    """Differentiable Airy formula for Fizeau fringe simulation.

    Implements:
        phi_scaled = global_scale * phi + tilt_x * X + tilt_y * Y
        I = I_max / (1 + F * sin(phi_scaled / 2) ** 2)

    where F = 4R / (1 - R)^2 is the coefficient of finesse.

    Learnable parameters (when learnable=True):
        - global_scale : multiplicative phase scaling (init 10.0)
        - reflectivity : mirror reflectance R via sigmoid (init 0.5)
        - I_max        : peak intensity
        - tilt_x, tilt_y : linear tilt coefficients
    """

    def __init__(
        self,
        R: float = 0.5,
        I_max: float = 1.0,
        global_scale: float = 1.0,
        learnable: bool = True,
        size: int = 128,
    ):
        super().__init__()

        if learnable:
            self.R_raw        = nn.Parameter(torch.tensor(R, dtype=torch.float32))
            self.I_max        = nn.Parameter(torch.tensor(I_max, dtype=torch.float32))
            self.global_scale = nn.Parameter(torch.tensor(global_scale, dtype=torch.float32))
            self.tilt_x       = nn.Parameter(torch.zeros(1))
            self.tilt_y       = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("R_raw",        torch.tensor(R, dtype=torch.float32))
            self.register_buffer("I_max",        torch.tensor(I_max, dtype=torch.float32))
            self.register_buffer("global_scale", torch.tensor(global_scale, dtype=torch.float32))
            self.register_buffer("tilt_x",       torch.zeros(1))
            self.register_buffer("tilt_y",       torch.zeros(1))

        # Normalised coordinate grids for tilt  (registered as buffer)
        lin = torch.linspace(-1, 1, size)
        gy, gx = torch.meshgrid(lin, lin, indexing='ij')
        self.register_buffer("grid_x", gx.clone())      # (H, W)
        self.register_buffer("grid_y", gy.clone())

    @property
    def R(self) -> torch.Tensor:
        """Reflectance constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self.R_raw)

    @property
    def F(self) -> torch.Tensor:
        """Coefficient of finesse: F = 4R / (1 - R)^2."""
        R = self.R
        return 4.0 * R / (1.0 - R) ** 2

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phi: Phase map (B, 1, H, W) in radians.
        Returns:
            Intensity map of the same shape.
        """
        # Scale phase to match real intensity magnitude
        phi_scaled = self.global_scale * phi

        # Add learnable tilt
        tilt = self.tilt_x * self.grid_x + self.tilt_y * self.grid_y  # (H, W)
        phi_scaled = phi_scaled + tilt

        return self.I_max / (1.0 + self.F * torch.sin(phi_scaled / 2.0) ** 2)
