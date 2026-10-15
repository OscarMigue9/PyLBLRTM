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


@_jit(inline="always", fastmath=True, error_model="numpy")
def _grid_index(base, step, k0, n, value):
    """Index k of the grid base + (k0 + k)·step just below *value*, clamped to [0, n]."""
    f = min(max((value - base) / step - k0, -1.0), n + 1.0)
    return min(max(int(math.floor(f)), 0), n)


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _add_lines(nu, centers, c, y, weight, cutoff, out):
    """out[i] += Σ_j weight_j · Re w(c_j·(nu_i − centre_j), y_j), straight into *out*,
    over the points with |nu_i − centre_j| ≤ cutoff.

    No profile matrix: the running sum stays in cache. *nu* must be a uniform
    grid. Per line, two passes with the same region test as :func:`_re_w`:

    1. every point within the cutoff, region I in closed real form, branch-free
       so it vectorises:
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

        # Points within the cutoff, [first, last): an estimate from the uniform
        # grid, then settled with the exact test |nu − centre| ≤ cutoff
        first = max(_grid_index(nu0, step, 0, n, cen - cutoff) - 2, 0)
        while first < n and cen - nu[first] > cutoff:
            first += 1
        last = min(_grid_index(nu0, step, 0, n, cen + cutoff) + 3, n)
        while last > first and nu[last - 1] - cen > cutoff:
            last -= 1

        # Views from index 0, so numba needs no negative-index check and vectorises
        nu_w, out_w = nu[first:last], out[first:last]
        for i in range(nu_w.shape[0]):
            x  = cj * (nu_w[i] - cen)
            x2 = x * x
            a  = p - x2
            far = wk * (p + x2) / (a * a + q * x2)
            out_w[i] += far if abs(x) >= x_far else 0.0

        if x_far > 0.0:
            # |nu − centre| < x_far / c, padded two points against rounding
            half = x_far / cj
            lo = min(max(_grid_index(nu0, step, 0, n, cen - half) - 2, first), last)
            hi = min(max(_grid_index(nu0, step, 0, n, cen + half) + 3, lo), last)
            wj = weight[j]
            for i in range(lo, hi):
                x = cj * (nu[i] - cen)
                if abs(x) < x_far:
                    out[i] += wj * _re_w(x, yj)


# ── Humlicek W4: far wings on coarser grids (wing_tolerance) ──────────────────
#
# Each line is split into rings with C² windows b(r; W) (1 up to W, smoothstep
# down to 0 at 2W):  φ = φ·b(W₁) + Σ φ·(b(W₂) − b(W₁)) + ... + φ·(1 − b(W_L)).
# The core φ·b(W₁) goes on the fine grid; ring l on a grid f_l = 4^l times
# coarser, with W_l = f_l·step/ratio so that every ring has the same D/W. Each
# coarse sum is interpolated to the fine grid (4-point Lagrange). The last ring
# carries the hard cutoff, a step that cannot be interpolated: near it the
# exact value replaces the interpolated one on the fine grid.

_LEVEL_FACTOR = 4          # each grid is 4× coarser than the previous one
_MAX_LEVELS   = 3          # 4, 16, 64 fine steps; coarser costs more at the cutoff than it saves


@_jit(inline="always", fastmath=True, error_model="numpy")
def _window(r, w, inv_w):
    """b(r; w): 1 up to w, C² smoothstep down to 0 at 2w. Branch-free and
    division-free (inv_w = 1/w > 0)."""
    t = min(max((r - w) * inv_w, 0.0), 1.0)
    return 1.0 - t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


@_jit(inline="always", fastmath=True, error_model="numpy")
def _span_window(out, k_start, k_end, base, step, k0, cen, cj, wk, p, q, x_far,
                 r_in, w_in, inv_in, w_out, inv_out):
    """Pass 1 of a ring on out[k_start:k_end]: region I closed form times the
    ring window. Where r < r_in = 2·w_in only the inner window is in transition
    (b_out = 1, as w_out ≥ 2·w_in); beyond, only the outer one: one per point."""
    out_w = out[k_start:k_end]           # from index 0: no negative-index check
    kf = k0 + k_start
    for k in range(out_w.shape[0]):
        d  = base + (kf + k) * step - cen
        r  = abs(d)
        near = r < r_in
        s  = _window(r, w_in if near else w_out, inv_in if near else inv_out)
        g  = 1.0 - s if near else s
        x  = cj * d
        x2 = x * x
        a  = p - x2
        far = wk * (p + x2) / (a * a + q * x2)
        out_w[k] += (far if abs(x) >= x_far else 0.0) * g


@_jit(inline="always", fastmath=True, error_model="numpy")
def _span_plateau(out, k_start, k_end, base, step, k0, cen, cj, wk, p, q, x_far):
    """Pass 1 of a ring where its window is exactly 1 (2·w_in ≤ r ≤ w_out)."""
    out_w = out[k_start:k_end]
    kf = k0 + k_start
    for k in range(out_w.shape[0]):
        x  = cj * (base + (kf + k) * step - cen)
        x2 = x * x
        a  = p - x2
        far = wk * (p + x2) / (a * a + q * x2)
        out_w[k] += far if abs(x) >= x_far else 0.0


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _add_ring(base, step, k0, n, centers, c, y, weight, w_in, w_out, cutoff, out):
    """out[k] += Σ_j weight_j·Re w(x)·(b(r; w_out) − b(r; w_in)) on the grid
    ν_k = base + (k0 + k)·step, k < n, where r = |ν_k − centre_j| ≤ cutoff.
    w_in ≤ 0 means no inner window (the ring is the core). Same two passes as
    :func:`_add_lines`: branch-free region I everywhere, then the core points.
    The plateau of each side, where the window is exactly 1, skips the window;
    it is taken one point inside its bounds, so no point there is misjudged."""
    w_in = w_in if w_in > 0.0 else 0.0
    r_in = 2.0 * w_in
    inv_in = 1.0 / w_in if w_in > 0.0 else 0.0
    inv_out = 1.0 / w_out
    r_max = min(2.0 * w_out, cutoff)
    pa = r_in + step                     # plateau, one step inside: pa ≤ r ≤ pb
    pb = min(w_out, r_max) - step

    for j in range(centers.shape[0]):
        cj, yj, cen, wj = c[j], y[j], centers[j], weight[j]
        wk = wj * _K1 * yj
        p  = 0.5 + yj * yj
        q  = 4.0 * yj * yj
        x_far = _REGION_I - yj

        first = max(_grid_index(base, step, k0, n, cen - r_max) - 2, 0)
        while first < n and cen - (base + (k0 + first) * step) > r_max:
            first += 1
        last = min(_grid_index(base, step, k0, n, cen + r_max) + 3, n)
        while last > first and (base + (k0 + last - 1) * step) - cen > r_max:
            last -= 1

        # [first, l0) window | [l0, l1) plateau | [l1, r0) window (centre) |
        # [r0, r1) plateau | [r1, last) window
        l0 = l1 = r0 = r1 = last
        if pb > pa:
            l0 = min(max(_grid_index(base, step, k0, n, cen - pb) + 1, first), last)
            l1 = min(max(_grid_index(base, step, k0, n, cen - pa) + 1, l0), last)
            r0 = min(max(_grid_index(base, step, k0, n, cen + pa) + 1, l1), last)
            r1 = min(max(_grid_index(base, step, k0, n, cen + pb) + 1, r0), last)
        _span_window(out, first, l0, base, step, k0, cen, cj, wk, p, q, x_far,
                     r_in, w_in, inv_in, w_out, inv_out)
        _span_plateau(out, l0, l1, base, step, k0, cen, cj, wk, p, q, x_far)
        _span_window(out, l1, r0, base, step, k0, cen, cj, wk, p, q, x_far,
                     r_in, w_in, inv_in, w_out, inv_out)
        _span_plateau(out, r0, r1, base, step, k0, cen, cj, wk, p, q, x_far)
        _span_window(out, r1, last, base, step, k0, cen, cj, wk, p, q, x_far,
                     r_in, w_in, inv_in, w_out, inv_out)

        if x_far > 0.0:
            half = x_far / cj
            lo = min(max(_grid_index(base, step, k0, n, cen - half) - 2, first), last)
            hi = min(max(_grid_index(base, step, k0, n, cen + half) + 3, lo), last)
            for k in range(lo, hi):
                d = base + (k0 + k) * step - cen
                x = cj * d
                if abs(x) < x_far:
                    r = abs(d)
                    near = r < r_in
                    s = _window(r, w_in if near else w_out, inv_in if near else inv_out)
                    out[k] += wj * _re_w(x, yj) * (1.0 - s if near else s)


def _lagrange_table(f: int) -> np.ndarray:
    """4-point Lagrange weights for the f fine points of a coarse interval,
    shape (4, f): row k multiplies the coarse value at offset k − 1."""
    t = np.arange(f) / f
    return np.ascontiguousarray(np.stack([
        -t * (t - 1.0) * (t - 2.0) / 6.0,
        (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0,
        -(t + 1.0) * t * (t - 2.0) / 2.0,
        (t + 1.0) * t * (t - 1.0) / 6.0,
    ]))


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _interp_add(coarse, m0, f, lag, i0, out):
    """out[i] += 4-point Lagrange of *coarse* (coarse[k] at coarse index m0 + k,
    every f fine steps) at fine index i0 + i, with the weights of *lag*
    (:func:`_lagrange_table`). One coarse interval at a time: fixed stencil
    values and contiguous weights, so the inner loop vectorises."""
    n = out.shape[0]
    for m in range(i0 // f, (i0 + n - 1) // f + 1):
        k = m - m0
        v0, v1, v2, v3 = coarse[k - 1], coarse[k], coarse[k + 1], coarse[k + 2]
        g0 = max(m * f, i0)
        g1 = min((m + 1) * f, i0 + n)
        ph = g0 - m * f
        out_w = out[g0 - i0:g1 - i0]
        l0, l1 = lag[0, ph:ph + out_w.shape[0]], lag[1, ph:ph + out_w.shape[0]]
        l2, l3 = lag[2, ph:ph + out_w.shape[0]], lag[3, ph:ph + out_w.shape[0]]
        for i in range(out_w.shape[0]):
            out_w[i] += v0 * l0[i] + v1 * l1[i] + v2 * l2[i] + v3 * l3[i]


@_jit(inline="always", fastmath=True, error_model="numpy")
def _ring_last_far(d, cj, wk, p, q, w_in, inv_in, cutoff):
    """Last ring at distance d from the centre, zero past the cutoff; region I
    (closed form), branch-free."""
    r  = abs(d)
    x  = cj * d
    x2 = x * x
    a  = p - x2
    v  = wk * (p + x2) / (a * a + q * x2) * (1.0 - _window(r, w_in, inv_in))
    return v if r <= cutoff else 0.0


@_jit(inline="always", fastmath=True, error_model="numpy")
def _ring_last(d, cj, yj, wj, wk, p, q, x_far, w_in, inv_in, cutoff):
    """Same as :func:`_ring_last_far` for any point, region I or not."""
    x = cj * d
    if abs(x) >= x_far:
        return _ring_last_far(d, cj, wk, p, q, w_in, inv_in, cutoff)
    r = abs(d)
    v = wj * _re_w(x, yj) * (1.0 - _window(r, w_in, inv_in))
    return v if r <= cutoff else 0.0


@_jit(cache=True, nogil=True, fastmath=True, error_model="numpy")
def _fix_cutoff(base, step, i0, n, f, lag, centers, c, y, weight, w_in, cutoff, side, out):
    """Near each line's cutoff (centre + side·cutoff), replace the interpolated
    last ring by its exact value: out[i] += exact − interpolated. Only where the
    4-point stencil straddles the step, i.e. within two coarse steps of it."""
    coarse_step = f * step
    reach = 2.0 * coarse_step + step
    buf = np.empty(16)
    inv_in = 1.0 / w_in

    for j in range(centers.shape[0]):
        cj, yj, cen, wj = c[j], y[j], centers[j], weight[j]
        wk = wj * _K1 * yj
        p  = 0.5 + yj * yj
        q  = 4.0 * yj * yj
        x_far = _REGION_I - yj
        far_only = cj * (cutoff - reach - coarse_step) >= x_far
        edge = cen + side * cutoff
        a = _grid_index(base, step, i0, n, edge - reach)
        b = min(_grid_index(base, step, i0, n, edge + reach) + 1, n)
        if b <= a:
            continue
        ma = (i0 + a) // f - 1
        mb = (i0 + b - 1) // f + 2
        for m in range(ma, mb + 1):
            buf[m - ma] = _ring_last(base + m * coarse_step - cen, cj, yj, wj, wk, p, q,
                                      x_far, w_in, inv_in, cutoff)
        for m in range((i0 + a) // f, (i0 + b - 1) // f + 1):
            k = m - ma
            v0, v1, v2, v3 = buf[k - 1], buf[k], buf[k + 1], buf[k + 2]
            g0 = max(m * f, i0 + a)
            g1 = min((m + 1) * f, i0 + b)
            ph = g0 - m * f
            out_w = out[g0 - i0:g1 - i0]
            nw = out_w.shape[0]
            l0, l1, l2, l3 = lag[0, ph:ph + nw], lag[1, ph:ph + nw], lag[2, ph:ph + nw], lag[3, ph:ph + nw]
            if far_only:     # always, for physical widths: branch-free, vectorises
                for i in range(nw):
                    exact = _ring_last_far(base + (g0 + i) * step - cen, cj, wk, p, q,
                                           w_in, inv_in, cutoff)
                    out_w[i] += exact - (v0 * l0[i] + v1 * l1[i] + v2 * l2[i] + v3 * l3[i])
            else:
                for i in range(nw):
                    exact = _ring_last(base + (g0 + i) * step - cen, cj, yj, wj, wk, p, q,
                                       x_far, w_in, inv_in, cutoff)
                    out_w[i] += exact - (v0 * l0[i] + v1 * l1[i] + v2 * l2[i] + v3 * l3[i])


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
    cutoff: float,
    out: np.ndarray,
) -> None:
    """out += Σ strength·φ(nu) over the prepared lines, each one only where
    |nu − centre| ≤ cutoff.

    *nu* must be a uniform grid and *out* a contiguous float64 array of the same
    length (a row slice of the optical depth stack works).
    """
    _require_numba()
    cutoff = min(float(cutoff), 1e100)   # the kernels assume finite numbers (fastmath)
    centers, c, y, weight = lines
    j0 = int(np.searchsorted(centers, nu[0] - cutoff, side="left"))
    j1 = int(np.searchsorted(centers, nu[-1] + cutoff, side="right"))
    if j1 > j0:
        _add_lines(nu, centers[j0:j1], c[j0:j1], y[j0:j1], weight[j0:j1], cutoff, out)


def humlicek_levels(step: float, cutoff: float, tolerance: float) -> list:
    """Coarse grids for the far wings, as (f, w_in, w_out) per level: a grid every
    f fine steps carrying the ring w_in < r < 2·w_out (w_out = 1e100 on the last).

    Interpolating a ring sampled every D across a window of width W errs by about
    0.8·(D/W)³ of the optical depth at worst (measured; doubling D costs ~8×), so
    D/W is set from *tolerance*, shared among the levels. Empty when even the
    first coarse grid would not fit inside the cutoff: then nothing is gained.
    """
    ratio = min(max((tolerance / (0.8 * _MAX_LEVELS)) ** (1.0 / 3.0), 0.02), 0.25)
    rings = []
    for level in range(1, _MAX_LEVELS + 1):
        f = _LEVEL_FACTOR ** level
        w = f * step / ratio
        if 2.0 * w >= cutoff - 3.0 * f * step:    # the ring must close before the cutoff
            break
        rings.append((f, w))
    return [
        (f, w, rings[i + 1][1] if i + 1 < len(rings) else 1e100)
        for i, (f, w) in enumerate(rings)
    ]


def humlicek_add_nested(
    origin: float,
    step: float,
    i0: int,
    lines: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cutoff: float,
    levels: list,
    out: np.ndarray,
) -> None:
    """Like :func:`humlicek_add`, with the far wings on the coarser grids of
    :func:`humlicek_levels`. *out* holds the fine points origin + (i0 + i)·step;
    the coarse grids are aligned to *origin*, so the result does not depend on
    how the spectrum is split into tiles.
    """
    _require_numba()
    cutoff = min(float(cutoff), 1e100)   # the kernels assume finite numbers (fastmath)
    centers, c, y, weight = lines
    n = out.shape[0]
    lo = origin + i0 * step
    hi = origin + (i0 + n - 1) * step

    def near(a, b):
        j0 = int(np.searchsorted(centers, a, side="left"))
        j1 = int(np.searchsorted(centers, b, side="right"))
        return centers[j0:j1], c[j0:j1], y[j0:j1], weight[j0:j1]

    # Core ring on the fine grid
    w_core = levels[0][1]
    r = min(2.0 * w_core, cutoff)
    _add_ring(origin, step, i0, n, *near(lo - r, hi + r), 0.0, w_core, cutoff, out)

    # Far rings on their coarse grids, interpolated to the fine one
    for f, w_in, w_out in levels:
        coarse_step = f * step
        m0 = i0 // f - 1
        nc = (i0 + n - 1) // f + 2 - m0 + 1
        coarse = np.zeros(nc)
        r = min(2.0 * w_out, cutoff)
        _add_ring(origin, coarse_step, m0, nc,
                  *near(origin + m0 * coarse_step - r, origin + (m0 + nc - 1) * coarse_step + r),
                  w_in, w_out, cutoff, coarse)
        _interp_add(coarse, m0, f, _lagrange_tables(f), i0, out)

    # The hard cutoff sits in the last ring: exact there, on the fine grid
    f, w_in, _ = levels[-1]
    reach = 2.0 * f * step + step
    for side in (-1.0, 1.0):
        _fix_cutoff(origin, step, i0, n, f, _lagrange_tables(f),
                    *near(lo - reach - side * cutoff, hi + reach - side * cutoff),
                    w_in, cutoff, side, out)


_LAGRANGE: dict = {}


def _lagrange_tables(f: int) -> np.ndarray:
    """:func:`_lagrange_table`, built once per f (dict reads are thread-safe)."""
    lag = _LAGRANGE.get(f)
    if lag is None:
        lag = _LAGRANGE.setdefault(f, _lagrange_table(f))
    return lag


# ── Backend selection ─────────────────────────────────────────────────────────

def get_voigt(name: str) -> Callable:
    """Return the Voigt backend called *name* (one of :data:`VOIGT_BACKENDS`)."""
    if name == "wofz":
        return voigt_wofz
    if name == "humlicek":
        _require_numba()
        return voigt_humlicek
    raise ValueError(f"Unknown Voigt backend '{name}'; choose one of {VOIGT_BACKENDS}")
