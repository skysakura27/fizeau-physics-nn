"""Minimal working example of the Fizeau PINN."""

import torch
import sys
sys.path.insert(0, '../')

from src.config import Config
from src.utils import create_circular_mask, normalize_phase

def main():
    """Run minimal example."""
    print("🚀 Fizeau Physics-Informed Neural Network - Minimal Example")
    print("=" * 60)
    
    config = Config("../config/config.yaml")
    print(f"✓ Configuration loaded: {config}")
    
    mask = create_circular_mask(size=512, radius=0.45)
    print(f"✓ Circular pupil mask created: {mask.shape}")
    
    batch_size = 2
    interferogram = torch.randn(batch_size, 1, 512, 512)
    print(f"✓ Dummy interferogram created: {interferogram.shape}")
    
    phase = torch.randn(batch_size, 1, 512, 512) * 0.5
    phase_normalized = normalize_phase(phase)
    print(f"✓ Phase normalized to [-π, π]: {phase_normalized.shape}")
    
    print("=" * 60)
    print("✓ Minimal example completed successfully!")

if __name__ == "__main__":
    main()
