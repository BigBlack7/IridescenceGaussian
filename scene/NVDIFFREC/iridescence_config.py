"""Iridescent thin-film configuration helpers.

Edit the constants in this module to hardcode the default thin-film and
substrate parameters used during iridescent BRDF training. This keeps the
workflow close to the Blender presets: simply change the numbers below
before launching ``train.py`` to try a different material, without having
to plumb new command-line arguments through the pipeline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils.general_utils import inverse_sigmoid

# Parameter order used throughout the iridescent BRDF implementation.
IRIDESCENT_PARAM_ORDER = (
    "film_thickness",  # nm-equivalent optical path difference scale
    "eta2",            # index of refraction for the thin film
    "eta3",            # base medium index of refraction
    "kappa3",          # absorption coefficient of the base medium
    "strength",        # lerp weight between white specular and iridescent Fresnel
)

# Allowed ranges for each parameter. Values outside of these ranges will
# be clamped when constructing defaults or converting to the internal
# unconstrained representation.
IRIDESCENT_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "film_thickness": (0.0, 5.0),
    "eta2": (1.0, 3.5),
    "eta3": (1.0, 5.0),
    "kappa3": (0.0, 5.0),
    "strength": (0.0, 1.0),
}

# Default thin-film configuration. Update these numbers to change the
# baseline material used when training on Blender datasets. They are
# interpreted in the same order as ``IRIDESCENT_PARAM_ORDER``.
DEFAULT_IRIDESCENT_PARAMS: Dict[str, float] = {
    "film_thickness": 0.6,
    "eta2": 1.8,
    "eta3": 2.7,
    "kappa3": 0.2,
    "strength": 0.9,
}

# Toggle per-parameter learning. Set a value to ``False`` to freeze the
# corresponding attribute at ``DEFAULT_IRIDESCENT_PARAMS[name]`` while
# still allowing other parameters to adapt during training.
IRIDESCENT_LEARNABLE: Dict[str, bool] = {
    "film_thickness": True,
    "eta2": True,
    "eta3": True,
    "kappa3": True,
    "strength": True,
}

# Small epsilon to keep sigmoid inverses numerically stable.
_SIGMOID_EPS = 1.0e-4


def _clamp_to_range(name: str, value: float) -> float:
    lo, hi = IRIDESCENT_PARAM_RANGES[name]
    return float(min(max(value, lo), hi))


def default_param_value(name: str) -> float:
    """Return the clamped default value for a parameter."""
    return _clamp_to_range(name, DEFAULT_IRIDESCENT_PARAMS[name])


def default_param_tensor(name: str, reference: torch.Tensor) -> torch.Tensor:
    """Create a tensor filled with the default value matching ``reference``."""
    value = default_param_value(name)
    return torch.full_like(reference, value)


def build_specular_raw_init(num_gaussians: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct initial specular features that decode to the defaults."""
    raw_values = []
    for name in IRIDESCENT_PARAM_ORDER:
        default_value = default_param_value(name)
        if name == "strength":
            # Map from [0, 1] to the unconstrained sigmoid space.
            normalized = torch.clamp(
                torch.tensor(default_value, dtype=dtype, device=device),
                min=_SIGMOID_EPS,
                max=1.0 - _SIGMOID_EPS,
            )
        else:
            lo, hi = IRIDESCENT_PARAM_RANGES[name]
            normalized = (default_value - lo) / (hi - lo)
            normalized = torch.clamp(
                torch.tensor(normalized, dtype=dtype, device=device),
                min=_SIGMOID_EPS,
                max=1.0 - _SIGMOID_EPS,
            )
        raw_values.append(inverse_sigmoid(normalized))
    raw = torch.stack(raw_values, dim=0)
    return raw.unsqueeze(0).repeat(num_gaussians, 1)


def default_param_matrix(num_gaussians: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the default parameter matrix broadcast to ``num_gaussians``."""
    base = torch.tensor(
        [default_param_value(name) for name in IRIDESCENT_PARAM_ORDER],
        dtype=dtype,
        device=device,
    )
    return base.unsqueeze(0).repeat(num_gaussians, 1)


def iridescent_param_is_learnable(name: str) -> bool:
    """Check whether a specific parameter should be optimised."""
    return bool(IRIDESCENT_LEARNABLE.get(name, True))


def any_iridescent_param_learnable() -> bool:
    """Return ``True`` if at least one iridescent attribute is trainable."""
    return any(iridescent_param_is_learnable(name) for name in IRIDESCENT_PARAM_ORDER)
