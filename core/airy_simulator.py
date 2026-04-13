"""Differentiable Airy forward model and gradient operator."""

from typing import Dict

import torch
import torch.nn.functional as F


class AiryPhysicsModel:
    """
    Airy diffraction physics model for interferometry.

    Implements the physical forward model and gradient computation
    for multi-beam interference.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.wavelength = config.get("wavelength", 632.8e-9)  # 632.8nm
        self.finesse = config.get("finesse", 10)  # Cavity finesse

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
        F_val = self.finesse

        intensity = I_max / (1 + F_val * torch.sin(phi / 2) ** 2)

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
        dI_dphi = -finesse * torch.sin(phi) / (2 * (1 + finesse * torch.sin(phi / 2) ** 2) ** 2)

        # Predicted intensity
        I_pred = self.forward_model(phase).squeeze(1)

        # Gradient
        gradient = dI_dphi * (I_meas - I_pred)

        return gradient.unsqueeze(1)
