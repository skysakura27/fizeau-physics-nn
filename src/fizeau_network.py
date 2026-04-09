"""Main Fizeau Physics-Informed Neural Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

from .utils import DWTPreprocessor, ZernikeBasis, AiryPhysicsModel


class AiryPINBlock(nn.Module):
    """
    Airy Physics-Informed Neural Network Block.

    Each block performs two physical actions:
    A. Physics correction: Airy function gradient operator
    B. Residual cleaning: CNN proximal operator for non-physical residuals
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Physics model for action A
        self.physics_model = AiryPhysicsModel(config.get('physics', {}))

        # CNN proximal operator for action B
        channels = config.get('channels', 64)
        self.proximal_net = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1)
        )

        # Learnable combination weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, phase: torch.Tensor, interferogram: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass of Airy-PIN block.

        Args:
            phase: Current phase estimate (B, 1, H, W)
            interferogram: Measured interferogram (B, 1, H, W)

        Returns:
            updated_phase: Updated phase after this block
            info: Dictionary with intermediate results
        """
        # Action A: Physics correction using Airy gradient
        physics_correction = self.physics_model.compute_gradient(phase, interferogram)

        # Action B: Residual cleaning using CNN proximal operator
        # Compute residual from previous iteration (simplified)
        residual_input = phase  # In practice, this would be phase - phase_prev
        neural_correction = self.proximal_net(residual_input)

        # Combine corrections
        total_correction = self.alpha * physics_correction + (1 - self.alpha) * neural_correction

        # Update phase
        updated_phase = phase + total_correction

        info = {
            'physics_correction': physics_correction,
            'neural_correction': neural_correction,
            'alpha': self.alpha,
            'total_correction': total_correction
        }

        return updated_phase, info


class ResidualDenoisingNet(nn.Module):
    """
    Residual denoising network for scatter noise suppression.

    Uses U-Net style architecture with multi-scale processing.
    """

    def __init__(self, config: Dict):
        super().__init__()
        channels = config.get('channels', 64)

        # Encoder
        self.enc1 = self._conv_block(1, channels)
        self.enc2 = self._conv_block(channels, channels * 2)
        self.enc3 = self._conv_block(channels * 2, channels * 4)

        # Decoder
        self.dec3 = self._conv_block(channels * 4, channels * 2)
        self.dec2 = self._conv_block(channels * 2, channels)
        self.dec1 = self._conv_block(channels, 1)

        # Skip connections
        self.skip2 = nn.Conv2d(channels * 2, channels * 2, 1)
        self.skip1 = nn.Conv2d(channels, channels, 1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # Decoder with skip connections
        d3 = self.dec3(e3)
        d3 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
        d3 = d3 + self.skip2(e2)

        d2 = self.dec2(d3)
        d2 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False)
        d2 = d2 + self.skip1(e1)

        d1 = self.dec1(d2)

        return d1


class ZernikeSupervisor(nn.Module):
    """
    Zernike Supervisor for hard constraint output.

    Predicts 36-order Zernike polynomial coefficients and reconstructs phase.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        num_modes = config.get('num_zernike_modes', 36)

        # Zernike basis
        self.zernike_basis = ZernikeBasis(num_modes)

        # Coefficient predictor network
        self.coeff_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes)
        )

    def forward(self, phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict Zernike coefficients and reconstruct phase.

        Args:
            phase: Input phase map (B, 1, H, W)

        Returns:
            reconstructed_phase: Phase reconstructed from coefficients (B, 1, H, W)
            coefficients: Zernike coefficients (B, num_modes)
        """
        # Predict coefficients
        coefficients = self.coeff_predictor(phase)

        # Reconstruct phase from coefficients
        reconstructed_phase = self.zernike_basis.reconstruct(coefficients)

        return reconstructed_phase, coefficients


class FizeauPhysicsNet(nn.Module):
    """
    Complete Fizeau Physics-Informed Neural Network.

    Four-stage architecture:
    1. Physical signal acquisition and deconstruction
    2. Frequency domain decoupling preprocessing (DWT)
    3. Physics-embedded core engine (Airy-PIN blocks)
    4. Topological hard constraint output (Zernike supervisor)
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Stage 2: DWT preprocessing (frequency domain decoupling)
        self.dwt_preprocessor = DWTPreprocessor(config.get('dwt', {}))

        # Stage 3: Airy-PIN iterative layers (physics-embedded core engine)
        num_blocks = config.get('num_unrolling_blocks', 5)
        self.unrolling_blocks = nn.ModuleList([
            AiryPINBlock(config.get('block', {})) for _ in range(num_blocks)
        ])

        # Residual denoising network (parallel branch)
        self.denoising_net = ResidualDenoisingNet(config.get('denoising', {}))

        # Stage 4: Zernike supervisor (topological hard constraint)
        self.zernike_supervisor = ZernikeSupervisor(config.get('zernike', {}))

        # Multi-scale fusion
        self.fusion_weights = nn.Parameter(torch.ones(num_blocks + 1))

    def forward(self, interferogram: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Complete forward pass through four stages.

        Args:
            interferogram: Input interferogram (B, 1, H, W)

        Returns:
            final_phase: Final reconstructed phase (B, 1, H, W)
            outputs: Dictionary with intermediate results from all stages
        """
        batch_size = interferogram.shape[0]
        outputs = {'interferogram': interferogram}

        # Stage 1: Physical signal acquisition and deconstruction
        # (Analysis of signal and noise sources - implemented in preprocessing)

        # Stage 2: Frequency domain decoupling preprocessing
        dwt_coeffs = self.dwt_preprocessor(interferogram)
        # Use LL (low-frequency) component as initial phase estimate
        phase = dwt_coeffs['LL']  # This is a simplification
        outputs['dwt_coeffs'] = dwt_coeffs
        outputs['initial_phase'] = phase

        # Stage 3: Physics-embedded core engine (Airy-PIN blocks)
        block_outputs = []
        for i, block in enumerate(self.unrolling_blocks):
            # Apply residual denoising
            denoised = self.denoising_net(phase)

            # Airy-PIN block
            phase, block_info = block(phase, interferogram)
            block_info['denoised'] = denoised
            block_outputs.append(block_info)

        outputs['block_outputs'] = block_outputs

        # Multi-scale fusion of block outputs
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        fused_phase = torch.zeros_like(phase)
        for i, weight in enumerate(fusion_weights[:-1]):
            fused_phase += weight * block_outputs[i]['total_correction']
        fused_phase += fusion_weights[-1] * phase

        # Stage 4: Topological hard constraint output
        final_phase, zernike_coeffs = self.zernike_supervisor(fused_phase)

        outputs['fused_phase'] = fused_phase
        outputs['zernike_coeffs'] = zernike_coeffs
        outputs['final_phase'] = final_phase

        return final_phase, outputs

    @staticmethod
    def load_from_checkpoint(checkpoint_path: str) -> 'FizeauPhysicsNet':
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        model = FizeauPhysicsNet(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_checkpoint(self, path: str, optimizer=None, epoch=None):
        """Save model checkpoint."""
        checkpoint = {
            'config': self.config,
            'model_state_dict': self.state_dict(),
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch

        torch.save(checkpoint, path)
