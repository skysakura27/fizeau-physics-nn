"""Configuration management for the Fizeau PINN."""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any

@dataclass
class Config:
    """Configuration class for the network."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.load()
    
    def load(self):
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.data.get(key, default)
    
    def __repr__(self) -> str:
        return f"Config({self.config_path})"

if __name__ == "__main__":
    config = Config()
    print(config)
