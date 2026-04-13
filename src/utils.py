"""Compatibility wrapper for legacy imports.

New implementations live under top-level `core` and `utils`.
"""

from core.dwt_ops import DWTPreprocessor, DWTPreprocessLayer
from core.zernike_logic import ZernikeBasis
from core.airy_simulator import AiryPhysicsModel
from utils.common import create_circular_mask, normalize_phase, compute_rms_error

__all__ = [
    "DWTPreprocessor",
    "DWTPreprocessLayer",
    "ZernikeBasis",
    "AiryPhysicsModel",
    "create_circular_mask",
    "normalize_phase",
    "compute_rms_error",
]
