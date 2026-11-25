"""Project AURA -- Volterra N-Body Engine.

A global waveform-relaxation orbital-mechanics engine that replaces step-by-step
numerical integration with damped Picard iteration of the Volterra integral form
of the equations of motion, including optional retarded (finite-``c``) gravity
and a reduced post-Newtonian term.
"""

from __future__ import annotations

from .operators import build_integral_operators, build_scheme
from .physics import (
    Body,
    PiecewiseVolterraStream,
    RelaxationResult,
    VolterraSystem,
    WindowResult,
    angular_momentum,
    eih_acceleration,
    linear_momentum,
    newtonian_energy,
)
from .presets import PRESET_NAMES, PRESETS, PresetConfig, get_preset

__version__ = "0.4.0"

__all__ = [
    "Body",
    "VolterraSystem",
    "RelaxationResult",
    "PiecewiseVolterraStream",
    "WindowResult",
    "newtonian_energy",
    "linear_momentum",
    "angular_momentum",
    "eih_acceleration",
    "build_integral_operators",
    "build_scheme",
    "PresetConfig",
    "PRESETS",
    "PRESET_NAMES",
    "get_preset",
    "__version__",
]
