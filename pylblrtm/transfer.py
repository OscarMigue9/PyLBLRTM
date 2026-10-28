"""
transfer.py — Thermal radiative transfer: the T, U and D terms.

Clear-sky radiance at the top of the atmosphere, for a surface of emissivity ε
at T_skin:

    L(nu) = ε·B(nu, T_skin)·T + U + (1 − ε)·T·D

  T   transmittance from the surface to the top along the view (1/mu_view)
  U   upwelling radiance the atmosphere emits at the top along the view
  D   downwelling radiance at the surface, along the diffusivity secant;
      it does not depend on the view angle

T, U and D do not depend on the surface. Layers are integrated with the
linear-in-tau scheme of LBLRTM (EMIN, xmerge.f90):

    tr  = exp(−dtau)
    f   = dtau/6                              dtau ≤ 1e-3
        = 1 − 2·(tr/(tr − 1) + 1/dtau)        dtau > 1e-3
    em  = (1 − tr)·(B_layer + (B_near − B_layer)·f)
    RAD = tr·RAD + em

B_near is the boundary towards the observer: the top of the layer for U, the
bottom for D. B_layer is the Planck radiance of the layer (layer_planck):

  "layer"       B(T_layer), the temperature the line strengths use. It is what
                LBLRTM does with user layers (IATM=0): B(TAVE).
  "density"     B(T̄), T̄ = ∫T dP / ΔP, the mass-weighted temperature, with T
                linear in ln P between the levels: what LBLATM averages.
  "arithmetic"  (B(T_top) + B(T_bottom)) / 2.

Radiances in W/(cm² sr cm⁻¹), as LBLRTM; × 1e7 gives mW/(m² sr cm⁻¹).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .atmosphere import AtmosphericProfile, HomogeneousPath
from .continuum import _check_h2o_continuum, h2o_continuum_optical_depth
from .forward import (
    _C2,
    _build_optical_depth,
    _check_wing_tolerance,
    _resolve_n_jobs,
)
from .spectral_data import SpectralData
from .voigt import VOIGT_BACKENDS

# ── Physical constants (universally fixed) ────────────────────────────────────
_C1 = 1.191042972e-12   # 2·h·c² [W cm² sr⁻¹], radiance per cm⁻¹ with nu in cm⁻¹

LAYER_PLANCK = ("layer", "density", "arithmetic")


def planck(nu: np.ndarray, T) -> np.ndarray:
    """Planck radiance B(nu, T) [W/(cm² sr cm⁻¹)]; nu in cm⁻¹, T in K (scalar or array)."""
    nu = np.asarray(nu, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)[..., None]
    return _C1 * nu ** 3 / np.expm1(_C2 * nu / T)


def _mass_weighted_temperature(p_hpa: np.ndarray, t_k: np.ndarray) -> np.ndarray:
    """∫T dP / ΔP per layer, with T linear in ln P between levels (TOA first)."""
    p1, p2, t1, t2 = p_hpa[:-1], p_hpa[1:], t_k[:-1], t_k[1:]
    return t1 + (t2 - t1) * (p2 / (p2 - p1) - 1.0 / np.log(p2 / p1))


def _layer_emission(dtau: np.ndarray, b_near: np.ndarray, b_layer: np.ndarray):
    """Emission and transmittance of one layer along the path (LBLRTM EMIN)."""
    absorbed = -np.expm1(-dtau)
    tr = np.exp(-dtau)
    f = dtau / 6.0
    thick = dtau > 1e-3
    f[thick] = 1.0 - 2.0 * (tr[thick] / (tr[thick] - 1.0) + 1.0 / dtau[thick])
    return absorbed * (b_layer + (b_near - b_layer) * f), tr


def compute_tud(
    spectral_data: SpectralData,
    profile: AtmosphericProfile,
    nu_min_cm1: float,
    nu_max_cm1: float,
    *,
    mu_view: float = 1.0,
    diffusivity: float = 1.67,
    layer_planck: str = "layer",
    step_cm1: float = 0.001,
    line_cutoff_cm1: float = 25.0,
    tile_width_cm1: float = 2.0,
    voigt: str = "wofz",
    n_jobs: int = 1,
    wing_tolerance: Optional[float] = None,
    h2o_continuum: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Monochromatic T, U and D of a vertical profile.

    Parameters
    ----------
    spectral_data, profile, nu_min_cm1, nu_max_cm1, step_cm1, line_cutoff_cm1,
    tile_width_cm1, voigt, n_jobs, wing_tolerance, h2o_continuum
        As in compute_transmittance. The profile must be vertical: a
        HomogeneousPath has no levels to integrate between.
    mu_view : float
        Cosine of the view zenith angle, for T and U. Default 1 (nadir).
    diffusivity : float
        Secant of the downwelling path, for D. Default 1.67.
    layer_planck : {"layer", "density", "arithmetic"}
        Planck radiance of each layer; see the module docstring. Default
        "layer", the one consistent with the temperature of the lines.

    Returns
    -------
    wavenumber_cm1, T, U, D : np.ndarray, shape (n_nu,)
        T is dimensionless; U and D in W/(cm² sr cm⁻¹).
    """
    if not isinstance(spectral_data, SpectralData):
        raise TypeError("spectral_data must be a SpectralData instance")
    if isinstance(profile, HomogeneousPath) or not isinstance(profile, AtmosphericProfile):
        raise TypeError("compute_tud needs a vertical AtmosphericProfile, not a HomogeneousPath")
    if nu_min_cm1 >= nu_max_cm1:
        raise ValueError("nu_min_cm1 must be strictly less than nu_max_cm1")
    if step_cm1 <= 0:
        raise ValueError("step_cm1 must be positive")
    if not (0.0 < mu_view <= 1.0):
        raise ValueError("mu_view must be in (0, 1]")
    if not (np.isfinite(diffusivity) and diffusivity >= 1.0):
        raise ValueError("diffusivity must be a secant, ≥ 1")
    if layer_planck not in LAYER_PLANCK:
        raise ValueError(f"layer_planck must be one of {LAYER_PLANCK}; got '{layer_planck}'")
    if voigt not in VOIGT_BACKENDS:
        raise ValueError(f"voigt must be one of {VOIGT_BACKENDS}; got '{voigt}'")
    _resolve_n_jobs(n_jobs)
    _check_wing_tolerance(wing_tolerance, voigt)

    n_pts = int(np.ceil((nu_max_cm1 - nu_min_cm1) / step_cm1)) + 1
    nu = nu_min_cm1 + np.arange(n_pts, dtype=np.float64) * step_cm1
    layers = profile.build_layers()
    if h2o_continuum:
        # before the line loop, which takes the time
        _check_h2o_continuum(nu, layers, line_cutoff_cm1)

    # Vertical optical depth per layer [n_layers, n_nu], TOA first
    od = _build_optical_depth(
        nu, spectral_data, layers, line_cutoff_cm1, tile_width_cm1,
        voigt=voigt, n_jobs=n_jobs, wing_tolerance=wing_tolerance,
    )
    if h2o_continuum:
        h2o_continuum_optical_depth(nu, layers, spectral_data, line_cutoff_cm1, add_to=od)

    t_lev = profile.temperature_k
    if layer_planck == "layer":
        t_layer = layers.temperature_k
    elif layer_planck == "density":
        t_layer = _mass_weighted_temperature(profile.pressure_hpa, t_lev)

    def b_layer(k: int) -> np.ndarray:
        if layer_planck == "arithmetic":
            return 0.5 * (planck(nu, t_lev[k]) + planck(nu, t_lev[k + 1]))
        return planck(nu, t_layer[k])

    # U: from the surface up, near boundary = top of each layer
    upwelling = np.zeros(nu.size)
    for k in reversed(range(layers.n_layers)):
        em, tr = _layer_emission(od[k] / mu_view, planck(nu, t_lev[k]), b_layer(k))
        upwelling = tr * upwelling + em

    # D: from the top down, near boundary = bottom of each layer
    downwelling = np.zeros(nu.size)
    for k in range(layers.n_layers):
        em, tr = _layer_emission(od[k] * diffusivity, planck(nu, t_lev[k + 1]), b_layer(k))
        downwelling = tr * downwelling + em

    transmittance = np.exp(-od.sum(axis=0) / mu_view)
    return nu, transmittance, upwelling, downwelling
