"""Planck radiance in mW m^-2 sr^-1 (cm^-1)^-1 and its inverse."""

from __future__ import annotations

import numpy as np

from photonix._validation import spectral_grid
from photonix.constants import FIRST_RADIATION_CONSTANT_MW, SECOND_RADIATION_CONSTANT_CM_K


def planck_radiance_nu(wavenumber_cm1, temperature_k) -> np.ndarray:
    """Return Planck radiance; scalar T gives [n_nu], vector T gives [n_T,n_nu]."""
    wavenumber = spectral_grid(wavenumber_cm1)
    temperature = np.asarray(temperature_k, dtype=np.float64)
    if temperature.ndim > 1 or not np.all(np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("temperature_k must be a positive finite scalar or vector")
    exponent = SECOND_RADIATION_CONSTANT_CM_K * wavenumber / temperature[..., None]
    with np.errstate(over="ignore"):
        return FIRST_RADIATION_CONSTANT_MW * wavenumber**3 / np.expm1(exponent)


def brightness_temperature_nu(wavenumber_cm1, radiance) -> np.ndarray:
    """Invert Planck radiance without silently clipping invalid radiance."""
    wavenumber = spectral_grid(wavenumber_cm1)
    values = np.asarray(radiance, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != wavenumber.size:
        raise ValueError("radiance last dimension must match wavenumber_cm1")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("radiance must be finite and positive to calculate brightness temperature")
    return (
        SECOND_RADIATION_CONSTANT_CM_K
        * wavenumber
        / np.log1p(FIRST_RADIATION_CONSTANT_MW * wavenumber**3 / values)
    )
