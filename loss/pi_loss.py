"""Physics-informed loss functions."""

from typing import Dict, Optional

import torch


def physics_informed_loss(
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    physics_residual: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """
    Placeholder physics-informed loss.

    Args:
        pred_phase: Predicted phase (B, 1, H, W)
        target_phase: Ground truth phase (B, 1, H, W)
        physics_residual: Optional physics residual term
        weights: Optional weights dict

    Returns:
        Scalar loss tensor
    """
    if weights is None:
        weights = {"reconstruction": 1.0, "physics": 1.0}

    recon_loss = torch.mean((pred_phase - target_phase) ** 2)
    physics_loss = torch.mean(physics_residual ** 2) if physics_residual is not None else torch.tensor(
        0.0, device=pred_phase.device, dtype=pred_phase.dtype
    )

    return weights["reconstruction"] * recon_loss + weights["physics"] * physics_loss
