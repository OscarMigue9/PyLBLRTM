"""
profiles.py — Reference atmospheric profiles for testing and validation.

Provides the US Standard Atmosphere 1976 (USSA76) with realistic VMR profiles
for the main IR-active gases, interpolated at standard pressure levels.

Reference VMR data:
  - Temperature/pressure: US Standard Atmosphere, 1976. NOAA/NASA/USAF.
  - H2O, O3, N2O, CH4, CO: AFGL Atmospheric Constituent Profiles (0-120 km),
    Anderson et al., 1986, AFGL-TR-86-0110 (Mid-Latitude Summer profile).
  - CO2: 415 ppm (post-2020 global mean, essentially uniform below 80 km).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .atmosphere import AtmosphericProfile

# ── US Standard Atmosphere 1976 ───────────────────────────────────────────────
# Standard pressure/temperature at key geopotential altitudes.
# Source: USSA 1976 official tables.
# Ordered surface → TOA (altitude ascending); reversed inside the function.

_USSA_ALT_KM = np.array([
     0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.,
    12., 15., 20., 25., 30., 35., 40., 45., 50.
])

_USSA_P_HPA = np.array([
    1013.25, 898.76, 794.95, 701.09, 616.40, 540.48, 471.81, 410.61,
     356.00, 307.42, 264.36, 226.32, 193.99, 121.12,  55.293,  25.492,
      11.970,   5.7459,  2.8714,   1.4910,   0.79779
])

_USSA_T_K = np.array([
    288.15, 281.65, 275.15, 268.66, 262.17, 255.68, 249.19, 242.70,
    236.22, 229.73, 223.25, 216.65, 216.65, 216.65, 216.65, 221.55,
    226.51, 236.51, 250.35, 264.16, 270.65
])

# ── AFGL Mid-Latitude Summer VMR profiles ────────────────────────────────────
# Defined at key altitudes (km); interpolated in log(VMR) vs altitude.
# Units: mol/mol.

_VMR_ALT_KM = np.array([
     0.,  1.,  2.,  3.,  4.,  5.,  7., 10., 12., 15., 20., 25., 30., 35., 40., 45., 50.
])

# H2O — strong exponential decrease through troposphere; ~3 ppm in stratosphere
_H2O_VMR = np.array([
    7.745e-3, 5.200e-3, 3.150e-3, 1.845e-3, 1.017e-3, 5.630e-4,
    1.300e-4, 3.000e-5, 9.000e-6, 3.500e-6, 2.900e-6, 3.200e-6,
    3.300e-6, 3.300e-6, 3.300e-6, 3.300e-6, 3.300e-6
])

# CO2 — essentially uniform below 50 km (415 ppm, modern value)
_CO2_VMR = np.full(_VMR_ALT_KM.size, 415.0e-6)

# O3 — maximum around 20-25 km (stratospheric ozone layer)
_O3_VMR = np.array([
    2.0e-8, 2.5e-8, 2.8e-8, 3.0e-8, 4.0e-8, 7.0e-8,
    1.2e-7, 1.8e-7, 2.6e-7, 8.0e-7, 5.0e-6, 8.0e-6,
    7.5e-6, 5.7e-6, 3.0e-6, 1.1e-6, 4.2e-7
])

# N2O — well-mixed in troposphere, photochemically destroyed in stratosphere
_N2O_VMR = np.array([
    3.20e-7, 3.20e-7, 3.20e-7, 3.20e-7, 3.20e-7, 3.20e-7,
    3.20e-7, 3.20e-7, 3.15e-7, 3.10e-7, 2.80e-7, 2.20e-7,
    1.50e-7, 7.50e-8, 2.50e-8, 6.00e-9, 1.00e-9
])

# CH4 — well-mixed in troposphere, oxidised in stratosphere
_CH4_VMR = np.array([
    1.800e-6, 1.800e-6, 1.800e-6, 1.800e-6, 1.800e-6, 1.800e-6,
    1.800e-6, 1.790e-6, 1.770e-6, 1.760e-6, 1.600e-6, 1.200e-6,
    7.500e-7, 3.500e-7, 1.700e-7, 7.000e-8, 3.000e-8
])

# CO — emitted at surface, increasing with altitude in stratosphere due to CO2 photolysis
_CO_VMR = np.array([
    1.50e-7, 1.20e-7, 1.00e-7, 8.0e-8, 7.0e-8, 6.0e-8,
    5.0e-8, 4.5e-8, 4.5e-8, 5.0e-8, 5.0e-8, 4.5e-8,
    5.0e-8, 6.5e-8, 9.5e-8, 1.45e-7, 2.00e-7
])

# Map gas name → (altitude_km array, VMR array)
_GAS_PROFILES: Dict[str, tuple] = {
    "H2O": (_VMR_ALT_KM, _H2O_VMR),
    "CO2": (_VMR_ALT_KM, _CO2_VMR),
    "O3":  (_VMR_ALT_KM, _O3_VMR),
    "N2O": (_VMR_ALT_KM, _N2O_VMR),
    "CH4": (_VMR_ALT_KM, _CH4_VMR),
    "CO":  (_VMR_ALT_KM, _CO_VMR),
}


def _log_interp(z_query: np.ndarray, z_ref: np.ndarray, vmr_ref: np.ndarray) -> np.ndarray:
    """Log-linear interpolation in VMR vs altitude (clamped at boundaries)."""
    log_vmr = np.log(np.clip(vmr_ref, 1e-30, None))
    log_q = np.interp(z_query, z_ref, log_vmr)
    return np.exp(log_q)


def us_standard_atmosphere(
    gases: Optional[List[str]] = None,
    alt_max_km: float = 50.0,
) -> AtmosphericProfile:
    """Build a US Standard Atmosphere 1976 profile with AFGL MLS gas VMRs.

    Parameters
    ----------
    gases : list of str, optional
        Gas names to include (HITRAN convention). Default: all six available
        gases ["H2O", "CO2", "O3", "N2O", "CH4", "CO"].
    alt_max_km : float
        Maximum altitude to include [km]. Default 50. Must be ≤ 50.

    Returns
    -------
    AtmosphericProfile
        Profile with ~21 levels from TOA down to surface, ready for
        ``compute_transmittance()``.

    Examples
    --------
    >>> from pylblrtm.profiles import us_standard_atmosphere
    >>> p = us_standard_atmosphere(gases=["CO2", "H2O"])
    >>> p
    AtmosphericProfile(...)
    """
    available = list(_GAS_PROFILES.keys())
    if gases is None:
        gases = available
    else:
        unknown = [g for g in gases if g not in _GAS_PROFILES]
        if unknown:
            raise ValueError(
                f"Unknown gas(es): {unknown}. "
                f"Available: {available}"
            )

    if alt_max_km > 50.0:
        raise ValueError("alt_max_km must be ≤ 50 km (USSA table limit)")

    # Select altitude levels up to alt_max_km
    keep = _USSA_ALT_KM <= alt_max_km + 1e-6
    alt  = _USSA_ALT_KM[keep]
    p_hpa = _USSA_P_HPA[keep]
    t_k   = _USSA_T_K[keep]

    # Interpolate VMR at selected altitude levels
    vmr: Dict[str, np.ndarray] = {}
    for gas in gases:
        z_ref, vmr_ref = _GAS_PROFILES[gas]
        vmr[gas] = _log_interp(alt, z_ref, vmr_ref)

    # AtmosphericProfile requires TOA-first order (pressure increasing)
    # Our tables are surface-first (altitude ascending), so reverse.
    return AtmosphericProfile(
        pressure_hpa=p_hpa[::-1].copy(),
        temperature_k=t_k[::-1].copy(),
        vmr={gas: v[::-1].copy() for gas, v in vmr.items()},
    )


def list_profile_gases() -> List[str]:
    """Return the gas names available in the standard profile library."""
    return list(_GAS_PROFILES.keys())
