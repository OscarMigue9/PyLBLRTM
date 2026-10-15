"""
voigt.py — Voigt line-shape backends for the forward model.

Every backend takes the same arguments and returns the same matrix:

    voigt(nu, centers, doppler, lorentz) -> phi  [n_nu × n_lines]

with Doppler and Lorentz half-widths (HWHM) in cm⁻¹ and each column normalised
to unit area over infinite support.

  wofz      scipy Faddeeva function. Exact reference; always available.
  humlicek  Humlicek W4 (1982) rational approximation compiled with numba.
            Needs the optional dependency:  pip install pylblrtm[fast]

The forward model uses wofz through that matrix (phi @ strength). Humlicek also
has a summing path, :func:`humlicek_prepare` + :func:`humlicek_add`, that adds
the lines straight into the optical depth without building phi: that is what
the forward model runs for voigt="humlicek". Everything releases the GIL, so
``n_jobs`` threads run in parallel with either backend.

The numba kernels compile on first use and are cached on disk afterwards.

Reference
---------
Humlicek, J. (1982). Optimized computation of the Voigt and complex
probability functions. JQSRT 27, 437-444.
"""

from __future__ import annotations

import cmath
import math
from typing import Callable, Tuple

import numpy as np
from scipy.special import wofz

try:
    import numba
except ImportError:      # optional: only voigt="humlicek" needs it
    numba = None

VOIGT_BACKENDS = ("wofz", "humlicek")

_SQRT_LN2 = math.sqrt(math.log(2.0))
_VOIGT_K  = _SQRT_LN2 / math.sqrt(math.pi)   # same expression as before, to the last bit


def _jit(**options):
    """numba.njit(**options), or the plain function when numba is missing
    (then it is never called: the Humlicek entry points raise first)."""
    return numba.njit(**options) if numba is not None else (lambda f: f)


def _require_numba() -> None:
    if numba is None:
        raise ImportError(
            "voigt='humlicek' needs numba. "
            "Install it with:  pip install pylblrtm[fast]"
        )


# ── wofz (reference) ──────────────────────────────────────────────────────────

def voigt_wofz(
    nu: np.ndarray,
    centers: np.ndarray,
    doppler: np.ndarray,
    lorentz: np.ndarray,
) -> np.ndarray:
    """Voigt profile matrix [n_nu × n_lines] via the Faddeeva function."""
    z = _SQRT_LN2 * ((nu[:, None] - centers) + 1j * lorentz) / doppler
    return wofz(z).real * (_VOIGT_K / doppler)


# ── Humlicek W4: numba kernels ────────────────────────────────────────────────

_K1       = 0.5641896      # region I/II coefficient (≈ 1/√π), Humlicek's Table 1
_REGION_I = 15.0           # |x| + y ≥ 15 → region I (far wings)


@_jit(inline="always", fastmath=True, error_model="numpy")
def _re_w(x, y):
    """Re w(x + iy) by Humlicek W4: four rational approximations over disjoint
    regions of the upper half-plane, coefficients verbatim from Table 1."""
    ax = abs(x)
    s  = ax + y
    T  = complex(y, -x)

    if s >= _REGION_I:
        # Region I — far wings
        w = T * _K1 / (0.5 + T * T)
    elif s >= 5.5:
        # Region II
        U = T * T
        w = T * (1.410474 + U * _K1) / (0.75 + U * (3.0 + U))
    elif y >= 0.195 * ax - 0.176:
        # Region III — moderate distance from line centre
        w = (
            16.4955
            + T * (20.20933
            + T * (11.96482
            + T * (3.778987
            + T *  0.5642236)))
        ) / (
            16.4955
            + T * (38.82363
            + T * (39.27121
            + T * (21.69274
            + T * (6.699398
            + T))))
        )
    else:
        # Region IV — near line centre
        U   = T * T
        num = T * (
            36183.31
            - U * (3321.9905
            - U * (1540.787
            - U * (219.0313
            - U * (35.76683
            - U * (1.320522
            - U *  0.56419)))))
        )
        den = (
            32066.6
            - U * (24322.84
            - U * (9022.228
            - U * (2186.181
            - U * (364.2191
            - U * (61.57037
            - U * (1.841439
            - U))))))
        )
        w = cmath.exp(U) - num / den

    return w.real


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _profile_matrix(nu, centers, inv_doppler, y, scale, out):
    """out[i, j] = Re w(x_ij, y_j) · scale_j, with x_ij = √ln2·(nu_i − centre_j)/α_j."""
    for i in range(nu.shape[0]):
        for j in range(centers.shape[0]):
            x = _SQRT_LN2 * (nu[i] - centers[j]) * inv_doppler[j]
            out[i, j] = _re_w(x, y[j]) * scale[j]


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _add_lines(nu, centers, c, y, weight, out):
    """out[i] += Σ_j weight_j · Re w(c_j·(nu_i − centre_j), y_j), straight into *out*.

    No profile matrix: the running sum stays in cache. *nu* must be a uniform
    grid. Per line, two passes with the same region test as :func:`_re_w`:

    1. every point, region I in closed real form, branch-free so it vectorises:
       Re[T·K1/(0.5 + T²)] = K1·y·(0.5 + y² + x²) / ((0.5 + y² − x²)² + 4y²x²),
       kept only where |x| + y ≥ 15 (a select, not a branch);
    2. the few points near the centre where |x| + y < 15, four-region Humlicek.
    """
    n = nu.shape[0]
    if n == 0:
        return
    nu0  = nu[0]
    step = (nu[n - 1] - nu0) / (n - 1) if n > 1 else 1.0

    for j in range(centers.shape[0]):
        cj, yj, cen = c[j], y[j], centers[j]
        wk = weight[j] * _K1 * yj
        p  = 0.5 + yj * yj
        q  = 4.0 * yj * yj
        x_far = _REGION_I - yj                         # region I ⇔ |x| ≥ 15 − y

        for i in range(n):
            x  = cj * (nu[i] - cen)
            x2 = x * x
            a  = p - x2
            far = wk * (p + x2) / (a * a + q * x2)
            out[i] += far if abs(x) >= x_far else 0.0

        if x_far > 0.0:
            # |nu − centre| < x_far / c, padded two points against rounding
            half = x_far / cj
            lo = min(max(int(math.floor((cen - half - nu0) / step)) - 2, 0), n)
            hi = min(max(int(math.ceil((cen + half - nu0) / step)) + 3, lo), n)
            wj = weight[j]
            for i in range(lo, hi):
                x = cj * (nu[i] - cen)
                if abs(x) < x_far:
                    out[i] += wj * _re_w(x, yj)


# ── Humlicek W4: public entry points ──────────────────────────────────────────

def voigt_humlicek(
    nu: np.ndarray,
    centers: np.ndarray,
    doppler: np.ndarray,
    lorentz: np.ndarray,
) -> np.ndarray:
    """Voigt profile matrix [n_nu × n_lines] via Humlicek W4 + numba."""
    _require_numba()
    nu          = np.ascontiguousarray(nu, dtype=np.float64)
    centers     = np.ascontiguousarray(centers, dtype=np.float64)
    inv_doppler = 1.0 / np.ascontiguousarray(doppler, dtype=np.float64)
    y           = _SQRT_LN2 * np.ascontiguousarray(lorentz, dtype=np.float64) * inv_doppler
    out         = np.empty((nu.size, centers.size), dtype=np.float64)
    _profile_matrix(nu, centers, inv_doppler, y, _VOIGT_K * inv_doppler, out)
    return out


def humlicek_prepare(
    shifted: np.ndarray,
    strength: np.ndarray,
    doppler: np.ndarray,
    lorentz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lines sorted by centre, with what :func:`humlicek_add` needs per line:
    (centre, √ln2/α, y = √ln2·γ/α, strength·√(ln2/π)/α)."""
    order = np.argsort(shifted, kind="stable")
    inv_doppler = 1.0 / doppler[order]
    return (
        np.ascontiguousarray(shifted[order]),
        _SQRT_LN2 * inv_doppler,
        _SQRT_LN2 * lorentz[order] * inv_doppler,
        strength[order] * _VOIGT_K * inv_doppler,
    )


def humlicek_add(
    nu: np.ndarray,
    lines: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    center_min: float,
    center_max: float,
    out: np.ndarray,
) -> None:
    """out += Σ strength·φ(nu) over the prepared lines centred in [center_min, center_max].

    *nu* must be a uniform grid and *out* a contiguous float64 array of the same
    length (a row slice of the optical depth stack works).
    """
    _require_numba()
    centers, c, y, weight = lines
    j0 = int(np.searchsorted(centers, center_min, side="left"))
    j1 = int(np.searchsorted(centers, center_max, side="right"))
    if j1 > j0:
        _add_lines(nu, centers[j0:j1], c[j0:j1], y[j0:j1], weight[j0:j1], out)


# ── Backend selection ─────────────────────────────────────────────────────────

def get_voigt(name: str) -> Callable:
    """Return the Voigt backend called *name* (one of :data:`VOIGT_BACKENDS`)."""
    if name == "wofz":
        return voigt_wofz
    if name == "humlicek":
        _require_numba()
        return voigt_humlicek
    raise ValueError(f"Unknown Voigt backend '{name}'; choose one of {VOIGT_BACKENDS}")
