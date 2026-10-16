"""Normalised Voigt profiles with Doppler and Lorentz HWHM in cm^-1."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy.special import wofz

_SQRT_LOG_2 = math.sqrt(math.log(2.0))
_SCALE = _SQRT_LOG_2 / math.sqrt(math.pi)


@njit(cache=True)
def _humlicek_real(x: float, y: float) -> float:
    """Humlicek W4 coefficients used by the production AER CUDA kernel."""
    t = complex(y, -x)
    s = abs(x) + y
    if s >= 15.0:
        w = t * 0.5641896 / (0.5 + t * t)
    elif s >= 5.5:
        u = t * t
        w = t * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
    elif y >= 0.195 * abs(x) - 0.176:
        w = (16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + t * 0.5642236)))) / (
            16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))
        )
    else:
        u = t * t
        numerator = t * (
            36183.31
            - u
            * (
                3321.9905
                - u * (1540.787 - u * (219.0313 - u * (35.76683 - u * (1.320522 - u * 0.56419))))
            )
        )
        denominator = 32066.6 - u * (
            24322.84
            - u * (9022.228 - u * (2186.181 - u * (364.2191 - u * (61.57037 - u * (1.841439 - u)))))
        )
        w = np.exp(u) - numerator / denominator
    return w.real


@njit(cache=True)
def _humlicek_profile(nu, center, doppler, lorentz):
    result = np.empty((nu.size, center.size), dtype=np.float64)
    for i in range(nu.size):
        for j in range(center.size):
            x = _SQRT_LOG_2 * (nu[i] - center[j]) / doppler[j]
            y = _SQRT_LOG_2 * lorentz[j] / doppler[j]
            result[i, j] = _humlicek_real(x, y) * _SCALE / doppler[j]
    return result


def voigt_profile(
    wavenumber_cm1,
    line_center_cm1,
    doppler_hwhm_cm1,
    lorentz_hwhm_cm1,
    *,
    method: str = "humlicek",
) -> np.ndarray:
    """Return a float64 [frequency, line] matrix, normalised over infinite support.

    Both backends use the same HWHM convention. ``scipy`` evaluates Faddeeva;
    ``humlicek`` evaluates the four-region approximation of the production recipe.
    Zero Lorentz width is allowed; Doppler widths must be strictly positive.
    """
    arrays = [
        np.ascontiguousarray(a, dtype=np.float64)
        for a in (wavenumber_cm1, line_center_cm1, doppler_hwhm_cm1, lorentz_hwhm_cm1)
    ]
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Voigt inputs must be finite one-dimensional arrays")
    nu, center, doppler, lorentz = arrays
    if not (center.shape == doppler.shape == lorentz.shape):
        raise ValueError("Line centers and widths must have the same shape")
    if np.any(doppler <= 0) or np.any(lorentz < 0):
        raise ValueError("Doppler HWHM must be positive and Lorentz HWHM nonnegative")
    if method == "humlicek":
        result = _humlicek_profile(nu, center, doppler, lorentz)
    elif method == "scipy":
        z = _SQRT_LOG_2 * ((nu[:, None] - center) + 1j * lorentz) / doppler
        result = wofz(z).real * (_SCALE / doppler)
    else:
        raise ValueError("voigt method must be 'humlicek' or 'scipy'")
    if not np.isfinite(result).all():
        raise FloatingPointError("Voigt evaluation produced a non-finite profile")
    return result
