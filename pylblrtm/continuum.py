"""
continuum.py — MT_CKD continuum absorption of H2O and CO2.

Port of ``photonix.spectroscopy.continuum`` (Forward_repo/packages/photonix),
the continuum of the thermal production recipe, with the same tables, formulas
and constants. The table ``data/continuum_mtckd.npz`` is byte-identical to
photonix's ``resources/aer/continuum.npz`` (SHA256 6ea56d7b…bcd):

  H2O  self and foreign coefficients, temperature exponent of the self part
       (LBLRTM data/absco-ref_wv-mt-ckd.nc, MT_CKD), 620–2780 cm⁻¹
  CO2  coefficient (AER-RC/LBLRTM src/contnm.f90, BLOCK DATA BFCO2, MT_CKD),
       0–10000 cm⁻¹, used below 2000 cm⁻¹ only: above it LBLRTM applies the
       χ-factor and band-head corrections, which are not part of this recipe.

    τ_H2O = (C_s(ν)·(T_ref/T)^n(ν)·x + C_f(ν)·(1 − x)) · (P/P_ref)(T_ref/T) · R(ν,T) · N_H2O
    τ_CO2 = C(ν) · 1e-20 · (P/P_ref)(T_ref/T) · R(ν,T) · N_CO2

with x the H2O volume mixing ratio, N the column [molec cm⁻²], P_ref = 1013.0 hPa,
T_ref = 296 K and R(ν,T) = ν·tanh(c₂ν/2T) the radiation term. C_s, C_f are
interpolated log-linearly, n and C linearly. O2/N2 collision-induced absorption
is not included (no implementation of it exists in the source recipe).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

_C2 = 1.4387752               # second radiation constant at runtime [cm K]
_P_STD_HPA = 1013.25          # atm → hPa
CO2_MAX_CM1 = 2000.0          # χ-factor / band-head domain starts here
CONTINUUM_GASES = ("H2O", "CO2")


@dataclass(frozen=True)
class ContinuumTables:
    """MT_CKD reference coefficients (see the module docstring)."""

    h2o_wavenumber_cm1: np.ndarray
    h2o_self: np.ndarray
    h2o_foreign: np.ndarray
    h2o_temperature_exponent: np.ndarray
    co2_wavenumber_cm1: np.ndarray
    co2_coefficient: np.ndarray
    h2o_reference_temperature_k: float = 296.0
    h2o_reference_pressure_hpa: float = 1013.0
    co2_reference_temperature_k: float = 296.0
    co2_reference_pressure_hpa: float = 1013.0

    @property
    def h2o_range_cm1(self) -> Tuple[float, float]:
        return float(self.h2o_wavenumber_cm1[0]), float(self.h2o_wavenumber_cm1[-1])

    @property
    def co2_range_cm1(self) -> Tuple[float, float]:
        return float(self.co2_wavenumber_cm1[0]), min(float(self.co2_wavenumber_cm1[-1]), CO2_MAX_CM1)

    @classmethod
    def load(cls, path: Optional[str | Path] = None) -> "ContinuumTables":
        """Numerical tables only (no pickles)."""
        resource = Path(path) if path is not None else files("pylblrtm").joinpath(
            "data/continuum_mtckd.npz")
        with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            return cls(
                h2o_wavenumber_cm1=np.array(data["h2o_NU_TABLE"], dtype=np.float64),
                h2o_self=np.array(data["h2o_CS_REF"], dtype=np.float64),
                h2o_foreign=np.array(data["h2o_CF_REF"], dtype=np.float64),
                h2o_temperature_exponent=np.array(data["h2o_TEXP_REF"], dtype=np.float64),
                co2_wavenumber_cm1=np.array(data["co2_nu"], dtype=np.float64),
                co2_coefficient=np.array(data["co2_S"], dtype=np.float64),
                h2o_reference_temperature_k=float(data["h2o_T_REF"]),
                h2o_reference_pressure_hpa=float(data["h2o_P_REF_HP"]),
            )


@lru_cache(maxsize=1)
def default_tables() -> ContinuumTables:
    return ContinuumTables.load()


def _radiation_term(nu: np.ndarray, T: float) -> np.ndarray:
    """ν·tanh(c₂ν/2T), with the same small/large-argument branches as photonix."""
    u = nu * (_C2 / T)
    out = np.empty_like(u)
    low, high = u <= 0.01, u > 10.0
    mid = ~(low | high)
    out[low] = 0.5 * u[low] * nu[low]
    out[high] = nu[high]
    out[mid] = nu[mid] * np.tanh(0.5 * u[mid])
    return out


def continuum_optical_depth(
    wavenumber_cm1: np.ndarray,
    layers,
    gases: Iterable[str] = CONTINUUM_GASES,
    tables: Optional[ContinuumTables] = None,
) -> Dict[str, np.ndarray]:
    """Continuum optical depth of each gas, {gas: [n_layers, n_nu]}.

    Parameters
    ----------
    wavenumber_cm1 : array (n_nu,)
        Spectral grid [cm⁻¹].
    layers : _Layers
        From ``AtmosphericProfile.build_layers()`` (or a HomogeneousPath).
    gases : iterable of str
        Gases to include, among ("H2O", "CO2"); one missing from the layers is
        skipped.
    tables : ContinuumTables, optional
        Default: the bundled MT_CKD tables.

    Outside a table's range (H2O 620–2780, CO2 below 2000 cm⁻¹) the gas
    adds no continuum: nothing is extrapolated. ``continuum_coverage`` tells
    which part of a grid is covered.
    """
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    if nu.ndim != 1 or nu.size == 0 or not np.isfinite(nu).all() or np.any(nu <= 0):
        raise ValueError("Continuum frequencies must be finite, positive and one-dimensional")
    tables = default_tables() if tables is None else tables
    unknown = set(gases) - set(CONTINUUM_GASES)
    if unknown:
        raise ValueError(f"No continuum for {sorted(unknown)}; available: {CONTINUUM_GASES}")

    n_layers = layers.n_layers
    out: Dict[str, np.ndarray] = {}
    for gas in gases:
        if gas not in layers.column_cm2:
            continue
        lo, hi = tables.h2o_range_cm1 if gas == "H2O" else tables.co2_range_cm1
        inside = (nu >= lo) & ((nu <= hi) if gas == "H2O" else (nu < hi))
        od = np.zeros((n_layers, nu.size), dtype=np.float64)
        out[gas] = od
        if not inside.any():
            continue
        v = nu[inside]
        if gas == "H2O":
            c_self = np.exp(np.interp(v, tables.h2o_wavenumber_cm1, np.log(tables.h2o_self)))
            c_foreign = np.exp(np.interp(v, tables.h2o_wavenumber_cm1, np.log(tables.h2o_foreign)))
            t_exp = np.interp(v, tables.h2o_wavenumber_cm1, tables.h2o_temperature_exponent)
        else:
            c_co2 = np.interp(v, tables.co2_wavenumber_cm1, tables.co2_coefficient)

        for k in range(n_layers):
            T = float(layers.temperature_k[k])
            P = float(layers.pressure_atm[k])
            column = float(layers.column_cm2[gas][k])
            if not (np.isfinite(T) and T > 0 and np.isfinite(P) and P > 0):
                raise ValueError(f"Layer {k}: continuum requires positive finite T and P")
            if not np.isfinite(column) or column < 0:
                raise ValueError(f"Layer {k}, {gas}: invalid continuum column density")
            if column == 0.0:
                continue
            radiation = _radiation_term(v, T)
            if gas == "H2O":
                x = float(layers.vmr["H2O"][k])
                t_ratio = tables.h2o_reference_temperature_k / T
                density = (P * _P_STD_HPA / tables.h2o_reference_pressure_hpa) * t_ratio
                od[k, inside] = ((c_self * t_ratio ** t_exp * x + c_foreign * (1.0 - x))
                                 * density * radiation * column)
            else:
                density = (P * _P_STD_HPA / tables.co2_reference_pressure_hpa
                           * tables.co2_reference_temperature_k / T)
                od[k, inside] = c_co2 * column * density * 1.0e-20 * radiation
            if not np.isfinite(od[k]).all() or np.any(od[k] < 0):
                raise FloatingPointError(f"Layer {k}, {gas}: invalid continuum optical depth")
    return out


def add_continuum(od_by_gas: Dict[str, np.ndarray], wavenumber_cm1: np.ndarray, layers,
                  logger=None) -> Dict[str, np.ndarray]:
    """Add the continuum of each gas in the layers to its optical depth (new dict).

    Warns once per gas when part of the grid lies outside its table.
    """
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    out = dict(od_by_gas)
    for gas, od in continuum_optical_depth(nu, layers).items():
        covered = continuum_coverage(nu, gas)
        if logger is not None and not covered.all():
            lo, hi = (default_tables().h2o_range_cm1 if gas == "H2O"
                      else default_tables().co2_range_cm1)
            logger.warning("%s continuum covers %.0f–%.0f cm⁻¹; %d of %d grid points "
                           "(%.2f–%.2f cm⁻¹ requested) get no continuum",
                           gas, lo, hi, int((~covered).sum()), nu.size, nu[0], nu[-1])
        out[gas] = out[gas] + od if gas in out else od
    return out


def continuum_coverage(wavenumber_cm1: np.ndarray, gas: str,
                       tables: Optional[ContinuumTables] = None) -> np.ndarray:
    """Boolean mask of the grid points where *gas* has a continuum."""
    tables = default_tables() if tables is None else tables
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    if gas == "H2O":
        lo, hi = tables.h2o_range_cm1
        return (nu >= lo) & (nu <= hi)
    if gas == "CO2":
        lo, hi = tables.co2_range_cm1
        return (nu >= lo) & (nu < hi)
    raise ValueError(f"No continuum for {gas!r}; available: {CONTINUUM_GASES}")
