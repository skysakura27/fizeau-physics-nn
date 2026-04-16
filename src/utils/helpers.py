"""Common helpers and configuration utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


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


@dataclass
class Config:
    """Configuration class for the network."""

    model_path: Optional[str] = "configs/model_cfg.yaml"
    physics_path: Optional[str] = "configs/physics_cfg.yaml"

    def __post_init__(self):
        self.model_path = Path(self.model_path) if self.model_path else None
        self.physics_path = Path(self.physics_path) if self.physics_path else None
        self.model = self._load_yaml(self.model_path) if self.model_path else {}
        self.physics = self._load_yaml(self.physics_path) if self.physics_path else {}

        # Merge for convenience
        self.data: Dict[str, Any] = {}
        if self.model:
            self.data.update(self.model)
        if self.physics:
            self.data["physics"] = self.physics.get("physics", self.physics)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)

    def __repr__(self) -> str:
        return f"Config(model={self.model_path}, physics={self.physics_path})"


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
    physics_loss = (
        torch.mean(physics_residual ** 2)
        if physics_residual is not None
        else torch.tensor(0.0, device=pred_phase.device, dtype=pred_phase.dtype)
    )

    return weights["reconstruction"] * recon_loss + weights["physics"] * physics_loss
