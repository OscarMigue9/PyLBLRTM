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
each line is inside the table. pylblrtm does not subtract it, so with the
continuum the pedestals of the H2O lines are counted twice.

The table, data/mt_ckd_h2o.npz, is AER's absco-ref_wv-mt-ckd.nc copied by
scripts/mt_ckd_h2o_table.py. MT_CKD is © AER and redistributed under its
terms; the notice travels in the .npz (key "copyright").
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from .atmosphere import _Layers

# ── Physical constants (universally fixed) ────────────────────────────────────
_C2        = 1.4387752   # second radiation constant [cm K], as in forward.py
_P_STD_HPA = 1013.25     # hPa per atm, as in atmosphere.py

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


def _check_h2o_continuum(nu: np.ndarray, layers: _Layers) -> None:
    """Raise before any work if the continuum cannot be computed."""
    if "H2O" not in layers.column_cm2:
        raise ValueError("h2o_continuum needs 'H2O' in the profile vmr")
    table_nu = _table()["nu"]
    if nu.min() < table_nu[0] or nu.max() > table_nu[-1]:
        raise ValueError(
            f"The MT_CKD H2O continuum covers {table_nu[0]:g}–{table_nu[-1]:g} cm⁻¹; "
            f"requested {nu.min():g}–{nu.max():g} cm⁻¹"
        )


def h2o_continuum_optical_depth(
    nu: np.ndarray,
    layers: _Layers,
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
    return od
