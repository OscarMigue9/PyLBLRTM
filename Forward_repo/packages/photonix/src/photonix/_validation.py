"""Small input contracts shared by the numerical runtime."""

from __future__ import annotations

import numpy as np


def finite_vector(value, name: str, *, minimum_size: int = 1) -> np.ndarray:
    """Return a finite float64 vector, rejecting unintended broadcasting."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_size:
        raise ValueError(f"{name} must be a vector with at least {minimum_size} entries")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def spectral_grid(value, *, uniform: bool = False) -> np.ndarray:
    """Validate positive, strictly increasing wavenumbers in cm^-1."""
    array = finite_vector(value, "wavenumber_cm1", minimum_size=2 if uniform else 1)
    differences = np.diff(array)
    if np.any(array <= 0) or np.any(differences <= 0):
        raise ValueError("wavenumber_cm1 must be positive and strictly increasing")
    if uniform and not np.allclose(differences, differences[0], rtol=1e-7, atol=1e-10):
        raise ValueError("wavenumber_cm1 must have uniform spacing")
    return array


def positive_scalar(value, name: str) -> float:
    """Validate one positive finite scalar."""
    if np.ndim(value) != 0:
        raise ValueError(f"{name} must be a scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def viewing_cosine(value: float) -> float:
    """Validate a nadir/upward viewing cosine in (0, 1]."""
    result = positive_scalar(value, "mu_view")
    if result > 1:
        raise ValueError("mu_view must satisfy 0 < mu_view <= 1")
    return result
