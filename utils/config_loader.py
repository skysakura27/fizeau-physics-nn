"""Configuration loader utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
