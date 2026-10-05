"""
forward.py — Line-by-line transmittance forward model.

Physics reference: AER line-by-line (LBLRTM JRAD=0 convention).
  - Voigt profile via Faddeeva function (scipy).
  - HITRAN line-strength temperature scaling with partition function ratio.
  - Radiation term nu·tanh(c₂·nu / 2T) applied at each frequency (JRAD=0).
  - No line mixing; no continuum (pure line absorption).

The only required external inputs are a SpectralData object (from from_hapi or
from_files) and an AtmosphericProfile. The spectral data carries all line
parameters and TIPS tables; the profile carries pressures, temperatures, and VMRs.
"""

from __future__ import annotations

import math
import warnings
from typing import Tuple

import numpy as np
from scipy.special import wofz

from .spectral_data import SpectralData
from .atmosphere import AtmosphericProfile, _Layers
from .hitran_manager import MOLECULE_IDS

# Reverse map: HITRAN molecule integer ID → gas name string
_MOL_ID_TO_NAME: dict[int, str] = {v: k for k, v in MOLECULE_IDS.items()}

# ── Physical constants (universally fixed) ────────────────────────────────────
_T_REF    = 296.0       # HITRAN reference temperature [K]
_C2       = 1.4387752   # second radiation constant [cm K]
_DOPPLER  = 3.5810e-7   # sqrt(2R / (Na·c²)) [cm⁻¹ / sqrt(K·mol·g⁻¹)]
_SQRT_LN2 = math.sqrt(math.log(2.0))
_VOIGT_K  = _SQRT_LN2 / math.sqrt(math.pi)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _radiation_term(nu: np.ndarray, T: float) -> np.ndarray:
    """JRAD=0 radiation factor: nu · tanh(c₂·nu / 2T).

    The line strength is divided by this term at the (shifted) line centre and
    multiplied by it at every output frequency, producing the LBLRTM JRAD=0
    convention where line wings scale as R(nu,T) / R(nu_c,T).
    """
    return nu * np.tanh(0.5 * _C2 * nu / T)


def _voigt(
    nu: np.ndarray,
    centers: np.ndarray,
    doppler: np.ndarray,
    lorentz: np.ndarray,
) -> np.ndarray:
    """Voigt profile matrix [n_nu × n_lines] via Faddeeva function.

    Units: [cm⁻¹]⁻¹ per line, normalised over infinite support.
    """
    z = _SQRT_LN2 * ((nu[:, None] - centers) + 1j * lorentz) / doppler
    return wofz(z).real * (_VOIGT_K / doppler)


# ── Public API ────────────────────────────────────────────────────────────────

def compute_transmittance(
    spectral_data: SpectralData,
    profile: AtmosphericProfile,
    nu_min_cm1: float,
    nu_max_cm1: float,
    *,
    step_cm1: float = 0.001,
    line_cutoff_cm1: float = 25.0,
    tile_width_cm1: float = 2.0,
    mu_view: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute monochromatic transmittance via line-by-line integration.

    Works for any molecule/isotopologue combination present in *spectral_data*,
    as long as the corresponding gas is defined in *profile.vmr* under its HITRAN
    name (e.g. "H2O", "CO2", "O3").

    Parameters
    ----------
    spectral_data : SpectralData
        Line parameters and TIPS tables, typically from ``from_hapi()``.
    profile : AtmosphericProfile
        Vertical profile of pressure [hPa], temperature [K], and gas VMRs.
        Must be ordered TOA → surface (pressure strictly increasing).
    nu_min_cm1, nu_max_cm1 : float
        Spectral range [cm⁻¹].
    step_cm1 : float
        Wavenumber grid spacing [cm⁻¹]. Default 0.001 cm⁻¹ (~0.3 GHz).
    line_cutoff_cm1 : float
        Wing truncation radius per line [cm⁻¹]. Default 25 cm⁻¹.
    tile_width_cm1 : float
        Width of each spectral tile for the inner Voigt loop [cm⁻¹]. Default 2.
        Larger tiles use more memory; smaller tiles do more overhead.
    mu_view : float
        Cosine of the viewing zenith angle. 1.0 = nadir (vertical). Default 1.0.

    Returns
    -------
    wavenumber_cm1 : np.ndarray, shape (n_nu,)
        Wavenumber grid [cm⁻¹].
    transmittance : np.ndarray, shape (n_nu,)
        Monochromatic transmittance in [0, 1] at each wavenumber.

    Notes
    -----
    - HITRAN line strengths encode terrestrial isotopic abundance, so the full
      column is used for each isotopologue without abundance weighting.
    - CO2 line mixing is not applied; the result is pure Voigt line-by-line.
    - No continuum absorption is included. For H₂O or CO₂ continua, the
      transmittance will be slightly too high in their respective windows.

    Examples
    --------
    >>> from pylblrtm import from_hapi
    >>> from pylblrtm.atmosphere import AtmosphericProfile
    >>> from pylblrtm.forward import compute_transmittance
    >>> sd = from_hapi("CO2", 700, 750, iso_id=1)
    >>> profile = AtmosphericProfile([1.0, 10.0, 100.0, 1013.25],
    ...                               [220., 240., 265., 288.],
    ...                               {"CO2": [4.15e-4]*4})
    >>> nu, T = compute_transmittance(sd, profile, 710, 740)
    """
    if not isinstance(spectral_data, SpectralData):
        raise TypeError("spectral_data must be a SpectralData instance")
    if not isinstance(profile, AtmosphericProfile):
        raise TypeError("profile must be an AtmosphericProfile instance")
    if nu_min_cm1 >= nu_max_cm1:
        raise ValueError("nu_min_cm1 must be strictly less than nu_max_cm1")
    if step_cm1 <= 0:
        raise ValueError("step_cm1 must be positive")
    if not (0.0 < mu_view <= 1.0):
        raise ValueError("mu_view must be in (0, 1]")

    # Build uniform wavenumber grid
    n_pts = int(np.ceil((nu_max_cm1 - nu_min_cm1) / step_cm1)) + 1
    nu = nu_min_cm1 + np.arange(n_pts, dtype=np.float64) * step_cm1

    # Build layer arrays from profile
    layers = profile.build_layers()

    # Optical depth stack [n_layers, n_nu]
    od = _build_optical_depth(nu, spectral_data, layers, line_cutoff_cm1, tile_width_cm1)

    # Column transmittance (slant path via 1/mu_view)
    transmittance = np.exp(-od.sum(axis=0) / mu_view)
    return nu, transmittance


# ── Internal line-by-line kernel ──────────────────────────────────────────────

def _build_optical_depth(
    nu: np.ndarray,
    sd: SpectralData,
    layers: _Layers,
    line_cutoff_cm1: float,
    tile_width_cm1: float,
) -> np.ndarray:
    """Return vertical optical depth stack [n_layers, n_nu].

    Loops: layer → (mol_id, iso_id) group → spectral tile → line block.
    The JRAD=0 radiation term is divided out at shifted line centres and
    multiplied back at every frequency after the tile loop, matching the
    LBLRTM convention.
    """
    n_layers = layers.n_layers
    n_nu = nu.size
    od = np.zeros((n_layers, n_nu), dtype=np.float64)

    step = float(nu[1] - nu[0]) if n_nu > 1 else 0.001
    tile_samples = max(1, int(np.floor(tile_width_cm1 / step + 1e-8)))

    # Pull line arrays out of SpectralData once
    nu0     = sd.lines["nu0"]       # line centres [cm⁻¹]
    S_ref   = sd.lines["Sref"]      # line strength at T_ref [cm⁻¹/(molec cm⁻²)]
    g_air   = sd.lines["g_air"]     # air-broadened HWHM [cm⁻¹ atm⁻¹]
    g_self  = sd.lines["g_self"]    # self-broadened HWHM [cm⁻¹ atm⁻¹]
    E_low   = sd.lines["Elow"]      # lower-state energy [cm⁻¹]
    n_air   = sd.lines["n_air"]     # temperature exponent
    delta   = sd.lines["shift"]     # pressure shift [cm⁻¹ atm⁻¹]
    mol_ids = sd.lines["mol_id"]    # HITRAN molecule ID
    iso_ids = sd.lines["iso_id"]    # HITRAN isotopologue ID

    # Per-line pressure shift bound for conservative band masking
    max_P_atm = float(layers.pressure_atm.max()) if n_layers > 0 else 1.0
    shift_bound = np.abs(delta) * max_P_atm   # [cm⁻¹], same length as nu0

    # Group lines by (mol_id, iso_id) and pre-compute layer-invariant quantities
    pairs = sorted(set(zip(mol_ids.tolist(), iso_ids.tolist())))
    prepared = []
    for mol_id, iso_id in pairs:
        gas_name = _MOL_ID_TO_NAME.get(int(mol_id))
        if gas_name is None:
            warnings.warn(
                f"mol_id={mol_id} is not in the HITRAN catalogue; lines skipped.",
                RuntimeWarning, stacklevel=3,
            )
            continue

        if gas_name not in layers.column_cm2:
            warnings.warn(
                f"Gas '{gas_name}' (mol_id={mol_id}, iso_id={iso_id}) is in the "
                f"spectral data but not in the atmospheric profile vmr; "
                f"its lines are skipped.",
                RuntimeWarning, stacklevel=3,
            )
            continue

        # Line mask: select only lines that can contribute to the requested band
        line_mask = (
            (mol_ids == mol_id)
            & (iso_ids == iso_id)
            & (nu0 >= nu[0] - line_cutoff_cm1 - shift_bound)
            & (nu0 <= nu[-1] + line_cutoff_cm1 + shift_bound)
        )
        if not line_mask.any():
            continue

        # Molar mass for Doppler width calculation
        key = (int(mol_id), int(iso_id))
        molar_mass = sd.molar_masses.get(key)
        if molar_mass is None:
            warnings.warn(
                f"No molar mass found for (mol_id={mol_id}, iso_id={iso_id}); "
                f"using dry-air mass as fallback.",
                RuntimeWarning, stacklevel=3,
            )
            molar_mass = 28.97

        # Q(T_ref)/Q(T_layer) for each layer — computed once per isotopologue
        q_ref = sd.Q(_T_REF, mol_id, iso_id)
        q_ratios = np.array(
            [q_ref / sd.Q(float(T), mol_id, iso_id) for T in layers.temperature_k],
            dtype=np.float64,
        )

        prepared.append((mol_id, iso_id, gas_name, line_mask, molar_mass, q_ratios))

    if not prepared:
        return od

    # ── Main loop: layers × isotopologues × tiles ──────────────────────────────
    for k in range(n_layers):
        T  = float(layers.temperature_k[k])   # layer temperature [K]
        P  = float(layers.pressure_atm[k])    # layer pressure [atm]

        for mol_id, iso_id, gas_name, line_mask, molar_mass, q_ratios in prepared:
            col = float(layers.column_cm2[gas_name][k])  # gas column [molec cm⁻²]
            if col == 0.0:
                continue

            vmr_gas = float(layers.vmr[gas_name][k])
            P_self = vmr_gas * P      # partial pressure of absorber [atm]
            P_air  = P - P_self       # partial pressure of air [atm]

            idx      = np.flatnonzero(line_mask)
            centers  = nu0[idx]                                  # [cm⁻¹]
            shifted  = centers + delta[idx] * P_air             # pressure-shifted centres

            # ── Line strength at temperature T ─────────────────────────────────
            # Standard HITRAN formula × column density × JRAD=0 normalization:
            #   S(T) = S_ref · (Q_ref/Q_T) · exp(-c₂·E"·(1/T - 1/T_ref))
            #         · (1 - exp(-c₂·nu₀/T)) / (1 - exp(-c₂·nu₀/T_ref))
            #         · N_col / radiation_term(nu_shifted, T)
            strength = (
                S_ref[idx]
                * q_ratios[k]
                * np.exp(-_C2 * E_low[idx] * (1.0 / T - 1.0 / _T_REF))
                * (-np.expm1(-_C2 * centers / T))
                / (-np.expm1(-_C2 * centers / _T_REF))
                * col
                / _radiation_term(shifted, T)
            )

            if not np.isfinite(strength).all() or np.any(strength < 0):
                raise FloatingPointError(
                    f"Layer {k}, gas '{gas_name}': non-finite or negative line strength"
                )

            # ── Lorentz HWHM [cm⁻¹] ────────────────────────────────────────────
            lorentz = (
                (_T_REF / T) ** n_air[idx]
                * (g_air[idx] * P_air + g_self[idx] * P_self)
            )

            # ── Doppler HWHM [cm⁻¹] ────────────────────────────────────────────
            doppler = centers * _DOPPLER * math.sqrt(T / molar_mass)

            # ── Tile loop (memory-bounded Voigt evaluation) ────────────────────
            for t0 in range(0, n_nu, tile_samples):
                t1   = min(t0 + tile_samples, n_nu)
                tile = nu[t0:t1]
                nearby = (
                    (shifted >= tile[0] - line_cutoff_cm1)
                    & (shifted <= tile[-1] + line_cutoff_cm1)
                )
                if not nearby.any():
                    continue
                nb = np.flatnonzero(nearby)
                # Block Voigt to limit peak memory usage (~256 lines at a time)
                for b0 in range(0, nb.size, 256):
                    b = nb[b0 : b0 + 256]
                    phi = _voigt(tile, shifted[b], doppler[b], lorentz[b])
                    od[k, t0:t1] += phi @ strength[b]

        # Apply JRAD=0 radiation term at layer temperature for all frequencies
        od[k] *= _radiation_term(nu, T)

        if not np.isfinite(od[k]).all():
            raise FloatingPointError(f"Layer {k}: non-finite optical depth after radiation term")

    return od
