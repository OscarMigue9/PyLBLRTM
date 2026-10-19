"""
atmosphere.py — Atmospheric profile and hydrostatic layer construction.

The profile is always ordered TOA → surface (pressure strictly increasing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Physical constants — universally fixed, never user-configurable
_AVOGADRO  = 6.02214086e23   # mol⁻¹
_GRAVITY   = 9.80665          # m s⁻²
_M_DRY_AIR = 0.0289644       # kg mol⁻¹  (dry air molar mass)
_P_STD_HPA = 1013.25          # hPa       (standard atmosphere pressure)


@dataclass
class _Layers:
    """Layer-averaged quantities derived from an AtmosphericProfile.

    All arrays have shape (n_layers,), ordered TOA → surface (index 0 = topmost).
    """

    pressure_atm: np.ndarray         # geometric-mean layer pressure [atm]
    temperature_k: np.ndarray        # arithmetic-mean layer temperature [K]
    vmr: Dict[str, np.ndarray]       # {gas: arithmetic-mean VMR [mol/mol]}
    column_cm2: Dict[str, np.ndarray]  # {gas: gas column density [molec cm⁻²]}

    @property
    def n_layers(self) -> int:
        return len(self.pressure_atm)


class AtmosphericProfile:
    """Vertical levels ordered TOA → surface (pressure strictly increasing).

    Parameters
    ----------
    pressure_hpa : array-like, shape (n_levels,)
        Pressure at each level [hPa]. Must be strictly increasing (lowest value = TOA).
    temperature_k : array-like, shape (n_levels,)
        Temperature at each level [K]. Must be > 0.
    vmr : dict of {gas_name: array-like}
        Volume mixing ratio [mol/mol] at each level for each gas.
        Gas names must follow the HITRAN convention: "H2O", "CO2", "O3",
        "N2O", "CH4", "CO", etc.

    Examples
    --------
    >>> import numpy as np
    >>> from pylblrtm.atmosphere import AtmosphericProfile
    >>> profile = AtmosphericProfile(
    ...     pressure_hpa=[10.0, 100.0, 1000.0],
    ...     temperature_k=[220.0, 260.0, 290.0],
    ...     vmr={"H2O": [1e-5, 1e-3, 8e-3]},
    ... )
    """

    def __init__(
        self,
        pressure_hpa,
        temperature_k,
        vmr: Dict[str, object],
    ) -> None:
        p = np.asarray(pressure_hpa, dtype=np.float64)
        t = np.asarray(temperature_k, dtype=np.float64)

        if p.ndim != 1 or p.size < 2:
            raise ValueError("pressure_hpa must be a 1-D array with at least 2 levels")
        if not np.all(np.isfinite(p)) or np.any(p <= 0):
            raise ValueError("pressure_hpa must be finite and positive")
        if np.any(np.diff(p) <= 0):
            raise ValueError(
                "pressure_hpa must be strictly increasing from TOA "
                "(smallest pressure = highest altitude = index 0)"
            )
        if t.shape != p.shape:
            raise ValueError("temperature_k must have the same length as pressure_hpa")
        if not np.all(np.isfinite(t)) or np.any(t <= 0):
            raise ValueError("temperature_k must be finite and positive")

        vmr_clean: Dict[str, np.ndarray] = {}
        for gas, values in vmr.items():
            v = np.asarray(values, dtype=np.float64)
            if v.shape != p.shape:
                raise ValueError(
                    f"vmr['{gas}'] must have the same length as pressure_hpa "
                    f"({p.size}), got {v.size}"
                )
            if not np.all(np.isfinite(v)) or np.any(v < 0) or np.any(v > 1):
                raise ValueError(f"vmr['{gas}'] must be finite and in [0, 1]")
            vmr_clean[gas] = v

        self.pressure_hpa = p
        self.temperature_k = t
        self.vmr = vmr_clean

    @property
    def n_levels(self) -> int:
        return self.pressure_hpa.size

    @property
    def n_layers(self) -> int:
        return self.n_levels - 1

    def build_layers(self) -> _Layers:
        """Build layer-averaged quantities via hydrostatic integration.

        Pressure: geometric mean of bounding levels.
        Temperature and VMR: arithmetic mean of bounding levels.
        Column density: hydrostatic formula — N = ΔP / (g × m_air).
        """
        p = self.pressure_hpa
        t = self.temperature_k

        # Geometric mean pressure [atm]
        p_layer = np.sqrt(p[:-1] * p[1:]) / _P_STD_HPA

        # Arithmetic mean temperature [K]
        t_layer = 0.5 * (t[:-1] + t[1:])

        # Arithmetic mean VMR per layer
        vmr_layer: Dict[str, np.ndarray] = {
            gas: 0.5 * (v[:-1] + v[1:]) for gas, v in self.vmr.items()
        }

        # Dry-air column density [molec cm⁻²] per layer
        # N_air = ΔP [Pa] × Na / (g × M_air) / 1e4  [m² → cm²]
        dp_pa = np.abs(np.diff(p)) * 100.0           # hPa → Pa
        col_factor = _AVOGADRO / (_GRAVITY * _M_DRY_AIR)  # molec J⁻¹ = molec m⁻² Pa⁻¹
        air_col_cm2 = col_factor * dp_pa / 1e4        # molec cm⁻²

        col_cm2: Dict[str, np.ndarray] = {
            gas: v_layer * air_col_cm2 for gas, v_layer in vmr_layer.items()
        }

        return _Layers(p_layer, t_layer, vmr_layer, col_cm2)

    def __repr__(self) -> str:
        gases = list(self.vmr.keys())
        if isinstance(self, HomogeneousPath):
            return (f"HomogeneousPath(L={self.path_length_m:g} m, "
                    f"P={self.pressure_hpa[0]:.4g} hPa, T={self.temperature_k[0]:.1f} K, "
                    f"vmr={ {g: float(v[0]) for g, v in self.vmr.items()} })")
        return (
            f"AtmosphericProfile("
            f"{self.n_levels} levels, "
            f"P=[{self.pressure_hpa[0]:.4g}–{self.pressure_hpa[-1]:.4g}] hPa, "
            f"T=[{self.temperature_k.min():.1f}–{self.temperature_k.max():.1f}] K, "
            f"gases={gases})"
        )


_BOLTZMANN = 1.380649e-23    # J K⁻¹


class HomogeneousPath(AtmosphericProfile):
    """A horizontal (standoff) path: one pressure, temperature and VMR per gas.

    The gas column is the number density times the path length,
    N = VMR · P / (k_B T) · L, instead of the hydrostatic column of a vertical
    profile. It works anywhere an AtmosphericProfile does (compute_transmittance,
    run_forward) as a single layer; keep mu_view = 1, since L is already the
    slant length.

    VMR is the total for the molecule: HITRAN line intensities already include
    the natural abundance of each isotopologue, so load every isotopologue
    (from_hapi(..., iso_id="all")) and give the molecule's concentration.

    Parameters
    ----------
    path_length_m : float
        Path length [m].
    vmr : dict of {gas_name: float}
        Volume mixing ratio [mol/mol] of each gas along the path
        (1 ppm = 1e-6). Gas names follow MOLECULE_IDS ("H2O", "SF6", ...).
    pressure_hpa : float
        Pressure along the path [hPa]. Default 1013.25 (sea level).
    temperature_k : float
        Temperature along the path [K]. Default 296 (HITRAN reference).

    Examples
    --------
    >>> path = HomogeneousPath(100.0, {"NH3": 1e-6})        # 1 ppm over 100 m
    >>> nu, T = compute_transmittance(sd, path, 900, 1000)
    """

    def __init__(
        self,
        path_length_m: float,
        vmr: Dict[str, float],
        pressure_hpa: float = _P_STD_HPA,
        temperature_k: float = 296.0,
    ) -> None:
        L = float(path_length_m)
        p = float(pressure_hpa)
        t = float(temperature_k)
        if not (np.isfinite(L) and L > 0):
            raise ValueError("path_length_m must be finite and positive")
        if not (np.isfinite(p) and p > 0):
            raise ValueError("pressure_hpa must be finite and positive")
        if not (np.isfinite(t) and t > 0):
            raise ValueError("temperature_k must be finite and positive")
        if not vmr:
            raise ValueError("vmr must name at least one gas")
        vmr_clean: Dict[str, np.ndarray] = {}
        for gas, value in vmr.items():
            if np.ndim(value) != 0:
                raise ValueError(f"vmr['{gas}'] must be a single value for a homogeneous path")
            v = float(value)
            if not (np.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"vmr['{gas}'] must be finite and in [0, 1]")
            vmr_clean[gas] = np.array([v])
        if sum(float(v[0]) for v in vmr_clean.values()) > 1.0:
            raise ValueError("the VMRs of all gases add up to more than 1")

        # Same attributes as AtmosphericProfile, with a single level
        self.pressure_hpa = np.array([p])
        self.temperature_k = np.array([t])
        self.vmr = vmr_clean
        self.path_length_m = L

    @property
    def n_layers(self) -> int:
        return 1

    def build_layers(self) -> _Layers:
        """One layer: N_gas = VMR · P / (k_B T) · L [molec cm⁻²]."""
        p_pa = self.pressure_hpa[0] * 100.0
        t = self.temperature_k[0]
        air_col_cm2 = p_pa / (_BOLTZMANN * t) * self.path_length_m / 1e4   # m⁻² → cm⁻²
        return _Layers(
            pressure_atm=np.array([self.pressure_hpa[0] / _P_STD_HPA]),
            temperature_k=self.temperature_k.copy(),
            vmr={gas: v.copy() for gas, v in self.vmr.items()},
            column_cm2={gas: v * air_col_cm2 for gas, v in self.vmr.items()},
        )
