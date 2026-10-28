"""
continuum.py — Water-vapour continuum (MT_CKD, self + foreign).

Physics reference: AER MT_CKD 4.3, formula of mt_ckd_h2o_module.f90
(Mlawer et al. 2012, doi:10.1098/rsta.2011.0295):

    rho_rat = (P / P_ref) · (T_ref / T)
    tau(nu) = [C_s(nu) · (T_ref/T)^n(nu) · x + C_f(nu) · (1 − x)]
              · rho_rat · R(nu, T) · N_H2O

  x       H2O volume mixing ratio of the layer
  N_H2O   H2O column of the layer [molec cm⁻²]
  R       radiation term nu·tanh(c₂·nu / 2T), the same as for the lines
  C_s, C_f, n   self and foreign coefficients [cm² molec⁻¹ (cm⁻¹)⁻¹] and the
                self temperature exponent, every 10 cm⁻¹ from −20 to 20 000 cm⁻¹,
                at P_ref = 1013 hPa and T_ref = 296 K

C_s and C_f are interpolated in log, n linearly. Outside the table it is an
error, never an extrapolation.

MT_CKD defines the continuum as the absorption that lines cut at 25 cm⁻¹,
*with their value at 25 cm⁻¹ subtracted*, do not explain: that pedestal under
each line is inside the table. The forward model keeps it in the lines, so,
given the lines, the pedestal of the H2O lines is subtracted here instead:

    pedestal(nu) = R(nu, T) · Σ_{|nu − c_i| ≤ 25} S_i / R(c_i, T) · γ_i / (π (25² + γ_i²))

with c_i, S_i and γ_i the shifted centre, strength (with column) and Lorentz
HWHM of the forward model itself (forward._layer_line_terms). Lines plus
continuum then count it once, as LBLRTM does. Without lines, MT_CKD alone.

The table, data/mt_ckd_h2o.npz, is AER's absco-ref_wv-mt-ckd.nc copied by
scripts/mt_ckd_h2o_table.py. MT_CKD is © AER and redistributed under its
terms; the notice travels in the .npz (key "copyright").
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .atmosphere import _Layers
from .hitran_manager import MOLECULE_IDS
from .spectral_data import SpectralData

# ── Physical constants (universally fixed) ────────────────────────────────────
_C2        = 1.4387752   # second radiation constant [cm K], as in forward.py
_P_STD_HPA = 1013.25     # hPa per atm, as in atmosphere.py

_MT_CKD_CUTOFF = 25.0    # line cutoff MT_CKD is defined for [cm⁻¹]
_TABLE_PATH = Path(__file__).resolve().parent / "data" / "mt_ckd_h2o.npz"


@lru_cache(maxsize=1)
def _table() -> dict:
    """MT_CKD coefficients, read once."""
    with np.load(_TABLE_PATH, allow_pickle=False) as d:
        return {
            "nu":          d["nu"],
            "log_self":    np.log(d["self"]),
            "log_foreign": np.log(d["foreign"]),
            "texp":        d["texp"],
            "T_ref":       float(d["T_ref"]),
            "P_ref_hpa":   float(d["P_ref_hpa"]),
        }


def _check_h2o_continuum(
    nu: np.ndarray, layers: _Layers, line_cutoff_cm1: Optional[float] = None,
) -> None:
    """Raise before any work if the continuum cannot be computed; warn on a cutoff ≠ 25."""
    if "H2O" not in layers.column_cm2:
        raise ValueError("h2o_continuum needs 'H2O' in the profile vmr")
    table_nu = _table()["nu"]
    if nu.min() < table_nu[0] or nu.max() > table_nu[-1]:
        raise ValueError(
            f"The MT_CKD H2O continuum covers {table_nu[0]:g}–{table_nu[-1]:g} cm⁻¹; "
            f"requested {nu.min():g}–{nu.max():g} cm⁻¹"
        )
    if line_cutoff_cm1 is not None and line_cutoff_cm1 != _MT_CKD_CUTOFF:
        warnings.warn(
            f"MT_CKD is defined for lines cut at {_MT_CKD_CUTOFF:g} cm⁻¹; with "
            f"line_cutoff_cm1={line_cutoff_cm1:g} lines and continuum do not add up "
            f"to the H2O absorption MT_CKD assumes.",
            RuntimeWarning, stacklevel=3,
        )


def _subtract_h2o_pedestal(
    nu: np.ndarray, layers: _Layers, sd: SpectralData, line_cutoff_cm1: float, od: np.ndarray,
) -> None:
    """Subtract from *od* the pedestal the forward model leaves under each H2O line."""
    from .forward import _T_REF, _layer_line_terms, _radiation_term   # forward imports this module

    h2o = np.flatnonzero(sd.lines["mol_id"] == MOLECULE_IDS["H2O"])
    if h2o.size == 0:
        return
    isos, of_iso = np.unique(sd.lines["iso_id"][h2o], return_inverse=True)
    mol = MOLECULE_IDS["H2O"]

    for k in range(layers.n_layers):
        col = float(layers.column_cm2["H2O"][k])
        if col == 0.0:
            continue
        T = float(layers.temperature_k[k])
        q_ratio = np.array([sd.Q(_T_REF, mol, i) / sd.Q(T, mol, i) for i in isos])[of_iso]
        _, shifted, strength, lorentz = _layer_line_terms(
            sd.lines, h2o, q_ratio, col, T,
            float(layers.pressure_atm[k]), float(layers.vmr["H2O"][k]),
        )
        # Lorentz value of each line at the cutoff, summed over the lines within it
        # of every frequency: a cumulative sum over the lines sorted by centre
        at_cutoff = strength * lorentz / (np.pi * (line_cutoff_cm1 ** 2 + lorentz ** 2))
        order = np.argsort(shifted)
        centres = shifted[order]
        cumulative = np.concatenate(([0.0], np.cumsum(at_cutoff[order])))
        lo = np.searchsorted(centres, nu - line_cutoff_cm1, "left")
        hi = np.searchsorted(centres, nu + line_cutoff_cm1, "right")
        od[k] -= _radiation_term(nu, T) * (cumulative[hi] - cumulative[lo])


def h2o_continuum_optical_depth(
    nu: np.ndarray,
    layers: _Layers,
    spectral_data: Optional[SpectralData] = None,
    line_cutoff_cm1: float = _MT_CKD_CUTOFF,
    add_to: Optional[np.ndarray] = None,
) -> np.ndarray:
    """MT_CKD water-vapour continuum optical depth per layer.

    Parameters
    ----------
    nu : np.ndarray, shape (n_nu,)
        Wavenumber grid [cm⁻¹], within −20…20 000 cm⁻¹.
    layers : _Layers
        From ``profile.build_layers()`` (AtmosphericProfile or HomogeneousPath);
        it must hold "H2O".
    spectral_data : SpectralData, optional
        The lines of the forward model. Given, the pedestal of its H2O lines
        (each line's value at the cutoff, within the cutoff) is subtracted, so
        that lines plus continuum count it once, as MT_CKD assumes; the result
        can then be negative where H2O lines are strong. None (default): MT_CKD
        alone.
    line_cutoff_cm1 : float
        Line cutoff of the forward model [cm⁻¹], for the pedestal. Default 25.
    add_to : np.ndarray, optional, shape (n_layers, n_nu)
        Add the continuum into this array in place, e.g. the line optical depth
        of the forward model, instead of allocating a new one.

    Returns
    -------
    np.ndarray, shape (n_layers, n_nu)
        Vertical optical depth of the continuum (*add_to* itself if given).
    """
    _check_h2o_continuum(nu, layers)
    t = _table()

    c_self    = np.exp(np.interp(nu, t["nu"], t["log_self"]))
    c_foreign = np.exp(np.interp(nu, t["nu"], t["log_foreign"]))
    texp      = np.interp(nu, t["nu"], t["texp"])

    od = np.zeros((layers.n_layers, nu.size)) if add_to is None else add_to
    for k in range(layers.n_layers):
        T = float(layers.temperature_k[k])
        x = float(layers.vmr["H2O"][k])
        col = float(layers.column_cm2["H2O"][k])
        if col == 0.0:
            continue
        rho_rat = (float(layers.pressure_atm[k]) * _P_STD_HPA / t["P_ref_hpa"]) * (t["T_ref"] / T)
        radiation = nu * np.tanh(0.5 * _C2 * nu / T)
        od[k] += (
            (c_self * (t["T_ref"] / T) ** texp * x + c_foreign * (1.0 - x))
            * rho_rat * radiation * col
        )
    if spectral_data is not None:
        _subtract_h2o_pedestal(nu, layers, spectral_data, line_cutoff_cm1, od)
    return od
