"""Synthetic data generation utilities."""

from typing import Dict

import torch

from ..core.airy_simulator import AiryPhysicsModel


def generate_interferogram(phase: torch.Tensor, physics_cfg: Dict) -> torch.Tensor:
    """
    Generate interferogram from phase using the Airy physics model.

    Args:
        phase: Phase map (B, 1, H, W)
        physics_cfg: Physics configuration dictionary

    Returns:
        interferogram: Simulated interferogram (B, 1, H, W)
    """
    model = AiryPhysicsModel(physics_cfg)
    return model.forward_model(phase)
