"""
forward.py — Line-by-line transmittance forward model.

Physics reference: AER line-by-line (LBLRTM JRAD=0 convention).
  - Voigt profile via the Faddeeva function: scipy wofz or Humlicek W4 (voigt.py).
  - HITRAN line-strength temperature scaling with partition function ratio.
  - Radiation term nu·tanh(c₂·nu / 2T) applied at each frequency (JRAD=0).
  - No line mixing. The only continuum is MT_CKD H2O, optional (continuum.py).

The only required external inputs are a SpectralData object (from from_hapi or
from_files) and an AtmosphericProfile. The spectral data carries all line
parameters and TIPS tables; the profile carries pressures, temperatures, and VMRs.
"""

from __future__ import annotations

import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np

from .spectral_data import SpectralData
from .atmosphere import AtmosphericProfile, _Layers
from .continuum import _check_h2o_continuum, h2o_continuum_optical_depth
from .hitran_manager import MOLECULE_IDS
from .voigt import (
    VOIGT_BACKENDS,
    get_voigt,
    humlicek_add,
    humlicek_add_nested,
    humlicek_levels,
    humlicek_prepare,
)

# Reverse map: HITRAN molecule integer ID → gas name string
_MOL_ID_TO_NAME: dict[int, str] = {v: k for k, v in MOLECULE_IDS.items()}

# ── Physical constants (universally fixed) ────────────────────────────────────
_T_REF    = 296.0       # HITRAN reference temperature [K]
_C2       = 1.4387752   # second radiation constant [cm K]
_DOPPLER  = 3.5810e-7   # sqrt(2R / (Na·c²)) [cm⁻¹ / sqrt(K·mol·g⁻¹)]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _radiation_term(nu: np.ndarray, T: float) -> np.ndarray:
    """JRAD=0 radiation factor: nu · tanh(c₂·nu / 2T).

    The line strength is divided by this term at the (shifted) line centre and
    multiplied by it at every output frequency, producing the LBLRTM JRAD=0
    convention where line wings scale as R(nu,T) / R(nu_c,T).
    """
    return nu * np.tanh(0.5 * _C2 * nu / T)


def _resolve_n_jobs(n_jobs: int) -> int:
    """Number of worker threads: *n_jobs* itself if ≥ 1, every core if -1."""
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or not (n_jobs == -1 or n_jobs >= 1)
    ):
        raise ValueError(f"n_jobs must be a positive integer or -1 (all cores); got {n_jobs!r}")
    return (os.cpu_count() or 1) if n_jobs == -1 else int(n_jobs)


def _check_wing_tolerance(wing_tolerance, voigt: str) -> None:
    """None, or a relative tolerance in (0, 1) together with voigt="humlicek"."""
    if wing_tolerance is None:
        return
    if (
        isinstance(wing_tolerance, bool)
        or not isinstance(wing_tolerance, (int, float, np.floating))
        or not (0.0 < wing_tolerance < 1.0)
    ):
        raise ValueError(f"wing_tolerance must be None or in (0, 1); got {wing_tolerance!r}")
    if voigt != "humlicek":
        raise ValueError(
            "wing_tolerance needs voigt='humlicek' (wofz is the exact reference)"
        )


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
    voigt: str = "wofz",
    n_jobs: int = 1,
    wing_tolerance: Optional[float] = None,
    h2o_continuum: bool = False,
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
        Wing truncation radius per line [cm⁻¹], applied at every frequency:
        a line adds nothing where |nu − centre| > cutoff. Default 25 cm⁻¹.
    tile_width_cm1 : float
        Width of each spectral tile for the inner Voigt loop [cm⁻¹]. Default 2.
        Larger tiles use more memory; smaller tiles do more overhead. It does
        not change the result.
    mu_view : float
        Cosine of the viewing zenith angle. 1.0 = nadir (vertical). Default 1.0.
    voigt : {"wofz", "humlicek"}
        Voigt line-shape backend. "wofz" (default) is scipy's exact Faddeeva
        function; "humlicek" is the Humlicek W4 approximation compiled with
        numba, summing the lines straight into the optical depth — much faster,
        within 1e-4 relative of wofz (needs ``pip install pylblrtm[fast]``).
    n_jobs : int
        Threads used in parallel. 1 (default) runs serially; -1 uses every core.
        Threads take the spectral tiles of every layer from one shared queue,
        so a single-layer standoff path uses them all just like a full column.
        The result does not depend on it: every value is summed exactly as in
        the serial loop.
    wing_tolerance : float, optional
        None (default) computes every line on the fine grid. A value such as
        1e-4 computes the far wings on grids 4, 16 and 64 times coarser and
        interpolates them, keeping the optical depth within about that relative
        error of the exact result (measured 3–10× below it) at several times
        the speed. The line cores and the 25 cm⁻¹ cutoff stay exact. Needs
        voigt="humlicek".
    h2o_continuum : bool
        Add the MT_CKD 4.3 water-vapour continuum (self + foreign) to the
        optical depth of every layer. Default False. Needs "H2O" in the
        profile and a range within −20…20 000 cm⁻¹. See continuum.py.

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
    - The only continuum is H₂O's, and only with h2o_continuum=True; CO₂ and
      the O₂/N₂ collision-induced ones are not included. MT_CKD assumes each
      line has its value at 25 cm⁻¹ subtracted (AER's pedestal), which
      pylblrtm does not do.

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
    if voigt not in VOIGT_BACKENDS:
        raise ValueError(f"voigt must be one of {VOIGT_BACKENDS}; got '{voigt}'")
    _resolve_n_jobs(n_jobs)
    _check_wing_tolerance(wing_tolerance, voigt)

    # Build uniform wavenumber grid
    n_pts = int(np.ceil((nu_max_cm1 - nu_min_cm1) / step_cm1)) + 1
    nu = nu_min_cm1 + np.arange(n_pts, dtype=np.float64) * step_cm1

    # Build layer arrays from profile
    layers = profile.build_layers()
    if h2o_continuum:
        _check_h2o_continuum(nu, layers)   # before the line loop, which takes the time

    # Optical depth stack [n_layers, n_nu]
    od = _build_optical_depth(
        nu, spectral_data, layers, line_cutoff_cm1, tile_width_cm1,
        voigt=voigt, n_jobs=n_jobs, wing_tolerance=wing_tolerance,
    )
    if h2o_continuum:
        h2o_continuum_optical_depth(nu, layers, add_to=od)

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
    voigt: str = "wofz",
    n_jobs: int = 1,
    wing_tolerance: Optional[float] = None,
) -> np.ndarray:
    """Return vertical optical depth stack [n_layers, n_nu].

    Loops: layer → (mol_id, iso_id) group → spectral tile → line block.
    The JRAD=0 radiation term is divided out at shifted line centres and
    multiplied back at every frequency after the tile loop, matching the
    LBLRTM convention.

    With n_jobs > 1 every (layer, spectral tile) pair is a task on a thread
    pool, so a single-layer standoff path keeps every thread as busy as a full
    column does. Tasks write disjoint slices of the stack, both Voigt backends
    release the GIL, and every slice is summed in the same order as the serial
    loop.
    """
    voigt_fn = get_voigt(voigt)
    # Humlicek adds each tile's lines straight into od, without the phi matrix
    summing = voigt == "humlicek"
    n_layers = layers.n_layers
    n_nu = nu.size
    od = np.zeros((n_layers, n_nu), dtype=np.float64)

    step = float(nu[1] - nu[0]) if n_nu > 1 else 0.001
    tile_samples = max(1, int(np.floor(tile_width_cm1 / step + 1e-8)))

    # Far wings on coarser grids (empty: no grid fits, everything stays fine)
    _check_wing_tolerance(wing_tolerance, voigt)
    levels = (
        humlicek_levels(step, line_cutoff_cm1, wing_tolerance)
        if wing_tolerance is not None and n_nu > 1 else []
    )
    if levels:
        # Every line within the cutoff visits every tile at a fixed cost, which
        # dominates when a tile holds few coarse points: give each tile at least
        # 128 points of the coarsest grid. No phi matrix here, so memory is no
        # concern, and the result does not depend on the tiling.
        tile_samples = max(tile_samples, 128 * levels[-1][0])
    tile_starts = range(0, n_nu, tile_samples)
    n_workers = min(_resolve_n_jobs(n_jobs), max(n_layers * len(tile_starts), 1))

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

    # ── Per layer: line parameters of every isotopologue ───────────────────────
    def _layer_lines(k: int) -> Tuple[float, list]:
        """Layer temperature and (shifted, strength, doppler, lorentz) per isotopologue."""
        T  = float(layers.temperature_k[k])   # layer temperature [K]
        P  = float(layers.pressure_atm[k])    # layer pressure [atm]
        groups = []

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

            lines = (shifted, strength, doppler, lorentz)
            groups.append(humlicek_prepare(*lines) if summing else lines)

        return T, groups

    # ── One tile of one layer (memory-bounded Voigt evaluation) ────────────────
    def _tile(k: int, groups: list, t0: int) -> None:
        t1   = min(t0 + tile_samples, n_nu)
        tile = nu[t0:t1]
        if summing:
            for lines in groups:
                if levels:
                    humlicek_add_nested(nu[0], step, t0, lines, line_cutoff_cm1,
                                        levels, od[k, t0:t1])
                else:
                    humlicek_add(tile, lines, line_cutoff_cm1, od[k, t0:t1])
            return
        for shifted, strength, doppler, lorentz in groups:
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
                phi = voigt_fn(tile, shifted[b], doppler[b], lorentz[b])
                # Cutoff at every frequency, not per tile: the result must not
                # depend on tile_width_cm1, which only bounds memory
                phi[np.abs(tile[:, None] - shifted[b]) > line_cutoff_cm1] = 0.0
                od[k, t0:t1] += phi @ strength[b]

    def _finish(k: int, T: float) -> None:
        # Apply JRAD=0 radiation term at layer temperature for all frequencies
        od[k] *= _radiation_term(nu, T)

        if not np.isfinite(od[k]).all():
            raise FloatingPointError(f"Layer {k}: non-finite optical depth after radiation term")

    def _layer(k: int) -> None:
        T, groups = _layer_lines(k)
        for t0 in tile_starts:
            _tile(k, groups, t0)
        _finish(k, T)

    # ── Main loop: serial, or one queue of (layer, tile) tasks on threads ──────
    if n_workers == 1:
        for k in range(n_layers):
            _layer(k)
        return od

    # Every tile of every layer is a task, queued layer after layer, so no
    # thread waits for another until the end, whatever the number of layers.
    # The first task of a layer prepares its lines and the last one to finish
    # applies the radiation term and frees them: only the layers in progress
    # hold their lines in memory.
    layer_locks = [threading.Lock() for _ in range(n_layers)]
    lines_of = [None] * n_layers
    tiles_left = [len(tile_starts)] * n_layers
    count_lock = threading.Lock()

    def _task(task: Tuple[int, int]) -> None:
        k, t0 = task
        with layer_locks[k]:
            if lines_of[k] is None:
                lines_of[k] = _layer_lines(k)
            T, groups = lines_of[k]
        _tile(k, groups, t0)
        with count_lock:
            tiles_left[k] -= 1
            last = tiles_left[k] == 0
        if last:
            lines_of[k] = None
            _finish(k, T)

    tasks = [(k, t0) for k in range(n_layers) for t0 in tile_starts]
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        try:
            list(pool.map(_task, tasks))   # list() re-raises worker errors
        except BaseException:
            pool.shutdown(cancel_futures=True)   # skip the tasks not yet started
            raise

    return od
