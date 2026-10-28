"""
config.py — YAML-driven forward model runner.

Separates *what to compute* (the YAML) from *the data* (SpectralData + profile).
The user edits forward_config.yaml and passes it to run_forward(); the spectral
data and atmospheric profile are always provided as Python objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .spectral_data import SpectralData
from .atmosphere import AtmosphericProfile, HomogeneousPath
from .binning import bin_by_wavelength
from .continuum import add_continuum
from .forward import _build_optical_depth, _check_wing_tolerance, _resolve_n_jobs
from .hitran_manager import DEFAULT_CACHE_DIR, MOLECULE_IDS
from .voigt import VOIGT_BACKENDS

logger = logging.getLogger(__name__)
_ID_TO_NAME = {v: k for k, v in MOLECULE_IDS.items()}


# ── Default values for every config key ──────────────────────────────────────
_DEFAULTS: dict = {
    "spectral": {
        "nu_min_cm1": 700.0,
        "nu_max_cm1": 750.0,
        "step_cm1": 0.001,
    },
    "calculation": {
        "mu_view": 1.0,
        "line_cutoff_cm1": 25.0,
        "tile_width_cm1": 2.0,
        "voigt": "wofz",
        "n_jobs": 1,
        "wing_tolerance": None,
        "continuum": False,         # MT_CKD continuum of H2O and CO2 (continuum.py)
    },
    # Standoff scenario for run_path(); run_forward() takes the profile instead
    "path": None,
    "output": {
        "transmittance": True,
        "optical_depth": False,
        "optical_depth_layers": False,
        "per_gas": False,           # transmittance of each gas on its own
        "bin_um": None,             # average over wavelength bins of this width [µm]
        "attenuation_db_m": False,  # A = -10/L · log10(T) [dB/m] (homogeneous path)
        "save": {
            "enabled": False,
            "path": "results/",
            "format": "npz",
            "prefix": "forward",
        },
    },
}


# ── Config loading & validation ───────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (base is not mutated)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(source: Union[str, Path, dict]) -> dict:
    """Load a forward model configuration from a YAML file or dict.

    Missing keys are filled from sensible defaults so the user only needs to
    specify what they want to change. Returns a fully resolved dict.

    Parameters
    ----------
    source : str, Path, or dict
        Path to a ``.yaml`` file, or an already-loaded dict (useful for
        programmatic calls without touching the filesystem).

    Examples
    --------
    >>> cfg = load_config("forward_config.yaml")
    >>> cfg = load_config({"spectral": {"nu_min_cm1": 710, "nu_max_cm1": 730}})
    """
    if isinstance(source, dict):
        raw = source
    else:
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required to load YAML config files. "
                "Install it with:  pip install pyyaml"
            ) from None
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

    cfg = _deep_merge(_DEFAULTS, raw)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    sp = cfg["spectral"]
    ca = cfg["calculation"]
    out = cfg["output"]

    if sp["nu_min_cm1"] >= sp["nu_max_cm1"]:
        raise ValueError("spectral.nu_min_cm1 must be strictly less than nu_max_cm1")
    if sp["step_cm1"] <= 0:
        raise ValueError("spectral.step_cm1 must be positive")
    if ca["line_cutoff_cm1"] <= 0:
        raise ValueError("calculation.line_cutoff_cm1 must be positive")
    if not (0.0 < ca["mu_view"] <= 1.0):
        raise ValueError("calculation.mu_view must be in (0, 1]")
    if ca["voigt"] not in VOIGT_BACKENDS:
        raise ValueError(
            f"calculation.voigt must be one of {VOIGT_BACKENDS}; got '{ca['voigt']}'"
        )
    try:
        _resolve_n_jobs(ca["n_jobs"])
    except ValueError:
        raise ValueError(
            f"calculation.n_jobs must be a positive integer or -1 (all cores); "
            f"got {ca['n_jobs']!r}"
        ) from None
    try:
        _check_wing_tolerance(ca["wing_tolerance"], ca["voigt"])
    except ValueError as err:
        raise ValueError(f"calculation.{err}") from None
    if not isinstance(ca["continuum"], bool):
        raise ValueError(f"calculation.continuum must be true or false; got {ca['continuum']!r}")

    requested = [out["transmittance"], out["optical_depth"], out["optical_depth_layers"],
                 out["per_gas"], out["attenuation_db_m"]]
    if not any(requested):
        raise ValueError(
            "At least one output must be enabled: transmittance, optical_depth, "
            "optical_depth_layers, per_gas or attenuation_db_m"
        )
    b = out["bin_um"]
    if b is not None and (isinstance(b, bool) or not isinstance(b, (int, float)) or b <= 0):
        raise ValueError(f"output.bin_um must be a positive width in µm or null; got {b!r}")

    if cfg["path"] is not None:
        _validate_path(cfg["path"])

    fmt = out["save"]["format"]
    if fmt not in {"npz", "csv", "txt"}:
        raise ValueError(
            f"output.save.format must be 'npz', 'csv', or 'txt'; got '{fmt}'"
        )


_PATH_DEFAULTS: dict = {
    "length_m": None,
    "pressure_hpa": 1013.25,
    "temperature_k": 296.0,
    "gases": None,
    "gas_units": "vmr",          # vmr (mol/mol) | ppm | atm (partial pressure)
    "isotopologues": "all",      # "all" | int | list of int
    "cache_dir": None,           # HITRAN cache; None → ~/.pylblrtm/hitran_cache
}
_GAS_UNITS = ("vmr", "ppm", "atm")


def _validate_path(pa: dict) -> None:
    if not isinstance(pa, dict):
        raise ValueError("path must be a mapping (length_m, pressure_hpa, temperature_k, gases…)")
    unknown = set(pa) - set(_PATH_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown key(s) in path: {sorted(unknown)}; "
                         f"valid: {sorted(_PATH_DEFAULTS)}")
    p = {**_PATH_DEFAULTS, **pa}
    for key in ("length_m", "pressure_hpa", "temperature_k"):
        v = p[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (v > 0):
            raise ValueError(f"path.{key} must be a positive number; got {v!r}")
    if not isinstance(p["gases"], dict) or not p["gases"]:
        raise ValueError("path.gases must map each gas to its concentration, e.g. {CO2: 4.2e-4}")
    if p["gas_units"] not in _GAS_UNITS:
        raise ValueError(f"path.gas_units must be one of {_GAS_UNITS}; got {p['gas_units']!r}")
    for gas, v in p["gases"].items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
            raise ValueError(f"path.gases.{gas} must be a number ≥ 0; got {v!r}")
    iso = p["isotopologues"]
    ok = (iso == "all" or (isinstance(iso, int) and not isinstance(iso, bool) and iso > 0)
          or (isinstance(iso, list) and iso and all(isinstance(i, int) and i > 0 for i in iso)))
    if not ok:
        raise ValueError(f"path.isotopologues must be 'all', a positive integer or a list; got {iso!r}")


def _path_vmr(pa: dict) -> dict:
    """Concentrations of path.gases as VMR (mol/mol)."""
    p = {**_PATH_DEFAULTS, **pa}
    units = p["gas_units"]
    p_atm = p["pressure_hpa"] / 1013.25
    factor = {"vmr": 1.0, "ppm": 1e-6, "atm": 1.0 / p_atm}[units]
    return {gas: float(v) * factor for gas, v in p["gases"].items()}


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class ForwardResult:
    """Output of :func:`run_forward`.

    Only the fields requested in the config are populated; the rest are ``None``.

    Attributes
    ----------
    wavenumber_cm1 : np.ndarray, shape (n_nu,)
        Wavenumber grid [cm⁻¹].
    transmittance : np.ndarray or None, shape (n_nu,)
        Monochromatic transmittance in [0, 1].
    optical_depth : np.ndarray or None, shape (n_nu,)
        Total column optical depth (sum over all layers).
    optical_depth_layers : np.ndarray or None, shape (n_layers, n_nu)
        Optical depth per atmospheric layer.
    transmittance_per_gas : dict or None
        {gas: transmittance (n_nu,)} of each gas alone (``output.per_gas``).
        Their product is the total transmittance.
    wavelength_um : np.ndarray or None, shape (n_bins,)
        Bin centres [µm], increasing (``output.bin_um``).
    transmittance_binned : np.ndarray or None, shape (n_bins,)
        Total transmittance averaged in each wavelength bin.
    transmittance_per_gas_binned : dict or None
        {gas: (n_bins,)} each gas averaged in each bin.
    attenuation_db_m : np.ndarray or None
        A = -10/L · log10(T) [dB/m] of the total, on the bins if ``bin_um`` is
        set, else on the wavenumber grid (``output.attenuation_db_m``).
    attenuation_per_gas_db_m : dict or None
        {gas: A [dB/m]} of each gas, when ``per_gas`` is also on.
    skipped_gases : dict
        {gas: reason} for gases of a run_path() scenario left out because
        HITRAN has no lines for them (their transmittance is 1).
    config : dict
        The fully resolved configuration used for this run.
    """

    wavenumber_cm1: np.ndarray
    config: dict
    transmittance: Optional[np.ndarray] = field(default=None)
    optical_depth: Optional[np.ndarray] = field(default=None)
    optical_depth_layers: Optional[np.ndarray] = field(default=None)
    transmittance_per_gas: Optional[Dict[str, np.ndarray]] = field(default=None)
    wavelength_um: Optional[np.ndarray] = field(default=None)
    transmittance_binned: Optional[np.ndarray] = field(default=None)
    transmittance_per_gas_binned: Optional[Dict[str, np.ndarray]] = field(default=None)
    attenuation_db_m: Optional[np.ndarray] = field(default=None)
    attenuation_per_gas_db_m: Optional[Dict[str, np.ndarray]] = field(default=None)
    skipped_gases: Dict[str, str] = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    def save(
        self,
        path: Optional[str] = None,
        fmt: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> Path:
        """Write outputs to disk.

        Parameters fall back to ``output.save.*`` values in the config when
        not provided. Returns the path of the file that was written.
        """
        sv = self.config["output"]["save"]
        return _write(
            self,
            path=path or sv["path"],
            fmt=fmt or sv["format"],
            prefix=prefix or sv["prefix"],
        )

    def __repr__(self) -> str:
        nu = self.wavenumber_cm1
        parts = []
        if self.transmittance is not None:
            parts.append("transmittance")
        if self.optical_depth is not None:
            parts.append("optical_depth")
        if self.optical_depth_layers is not None:
            shape = self.optical_depth_layers.shape
            parts.append(f"optical_depth_layers{list(shape)}")
        if self.transmittance_per_gas is not None:
            parts.append(f"per_gas{list(self.transmittance_per_gas)}")
        if self.wavelength_um is not None:
            parts.append(f"binned[{self.wavelength_um.size} × "
                         f"{self.config['output']['bin_um']} µm]")
        if self.attenuation_db_m is not None:
            parts.append("attenuation_db_m")
        return (
            f"ForwardResult("
            f"nu=[{nu[0]:.2f}, {nu[-1]:.2f}] cm-1 | "
            f"step={nu[1]-nu[0]:.4g} cm-1 | "
            f"outputs=[{', '.join(parts)}])"
        )


# ── Main runner ───────────────────────────────────────────────────────────────

def run_forward(
    spectral_data: SpectralData,
    profile: AtmosphericProfile,
    config: Union[str, Path, dict],
) -> ForwardResult:
    """Run the forward model using a YAML config file or dict.

    The config controls *what* to compute and *how* (spectral range, resolution,
    viewing angle, outputs). The spectral lines and atmospheric profile are
    always passed as Python objects — they are not part of the YAML.

    Parameters
    ----------
    spectral_data : SpectralData
        Line parameters and TIPS tables, typically from ``from_hapi()``.
    profile : AtmosphericProfile
        Atmospheric profile (levels ordered TOA → surface).
    config : str, Path, or dict
        Path to a ``.yaml`` config file, or a plain dict.  Missing keys are
        filled from defaults — a minimal config only needs ``spectral.nu_min_cm1``
        and ``spectral.nu_max_cm1``.

    Returns
    -------
    ForwardResult
        Container with the requested outputs. Call ``.save()`` to write to disk.

    Examples
    --------
    >>> from pylblrtm import from_hapi, run_forward, us_standard_atmosphere
    >>> sd      = from_hapi("CO2", 700, 750, iso_id=1)
    >>> profile = us_standard_atmosphere(gases=["CO2"])
    >>> result  = run_forward(sd, profile, "forward_config.yaml")
    >>> result.transmittance          # np.ndarray, shape (n_nu,)
    >>> result.save()                 # writes to output.save.path in the YAML
    """
    if not isinstance(spectral_data, SpectralData):
        raise TypeError("spectral_data must be a SpectralData instance")
    if not isinstance(profile, AtmosphericProfile):
        raise TypeError("profile must be an AtmosphericProfile instance")

    cfg = load_config(config)
    sp  = cfg["spectral"]
    ca  = cfg["calculation"]
    out = cfg["output"]

    # Build wavenumber grid
    nu_min = float(sp["nu_min_cm1"])
    nu_max = float(sp["nu_max_cm1"])
    step   = float(sp["step_cm1"])
    n_pts  = int(np.ceil((nu_max - nu_min) / step)) + 1
    nu     = nu_min + np.arange(n_pts, dtype=np.float64) * step

    # Build layer arrays
    layers = profile.build_layers()

    if out["attenuation_db_m"] and not isinstance(profile, HomogeneousPath):
        raise ValueError(
            "output.attenuation_db_m needs a HomogeneousPath (a path length L); "
            "a vertical profile has no single length"
        )

    def optical_depth(sd: SpectralData) -> np.ndarray:
        """Core line-by-line optical depth [n_layers, n_nu]."""
        return _build_optical_depth(
            nu,
            sd,
            layers,
            line_cutoff_cm1=float(ca["line_cutoff_cm1"]),
            tile_width_cm1=float(ca["tile_width_cm1"]),
            voigt=ca["voigt"],
            n_jobs=ca["n_jobs"],
            wing_tolerance=ca["wing_tolerance"],
        )

    mu = float(ca["mu_view"])

    # One run per gas when each gas is wanted on its own; τ adds up, so the
    # total is their sum
    od_gas: Dict[str, np.ndarray] = {}
    if out["per_gas"]:
        for mol_id in spectral_data.mol_ids:
            name = _ID_TO_NAME.get(mol_id, str(mol_id))
            if name in layers.column_cm2:
                od_gas[name] = optical_depth(spectral_data.for_molecule(mol_id))
        if ca["continuum"]:
            od_gas = add_continuum(od_gas, nu, layers, logger)
        od = sum(od_gas.values()) if od_gas else np.zeros((layers.n_layers, nu.size))
    else:
        od = optical_depth(spectral_data)
        if ca["continuum"]:
            od = od + sum(add_continuum({}, nu, layers, logger).values(), np.zeros_like(od))

    # Assemble requested outputs
    result = ForwardResult(wavenumber_cm1=nu, config=cfg)
    T_total = np.exp(-od.sum(axis=0) / mu)

    if out["transmittance"]:
        result.transmittance = T_total

    if out["optical_depth"]:
        result.optical_depth = od.sum(axis=0)

    if out["optical_depth_layers"]:
        result.optical_depth_layers = od

    if out["per_gas"]:
        result.transmittance_per_gas = {g: np.exp(-o.sum(axis=0) / mu) for g, o in od_gas.items()}

    # Wavelength bins: the mean of T in each bin (not exp of the mean τ)
    if out["bin_um"] is not None:
        result.wavelength_um, result.transmittance_binned = bin_by_wavelength(
            nu, T_total, float(out["bin_um"]))
        if out["per_gas"]:
            result.transmittance_per_gas_binned = {
                g: bin_by_wavelength(nu, T, float(out["bin_um"]))[1]
                for g, T in result.transmittance_per_gas.items()
            }

    if out["attenuation_db_m"]:
        L = profile.path_length_m

        def att(T):
            return -10.0 / L * np.log10(np.clip(T, 1e-300, 1.0))

        binned = out["bin_um"] is not None
        result.attenuation_db_m = att(result.transmittance_binned if binned else T_total)
        if out["per_gas"]:
            per = result.transmittance_per_gas_binned if binned else result.transmittance_per_gas
            result.attenuation_per_gas_db_m = {g: att(T) for g, T in per.items()}

    # Auto-save if the config requests it
    if out["save"]["enabled"]:
        result.save()

    return result


def run_path(config: Union[str, Path, dict]) -> ForwardResult:
    """Run a standoff scenario described entirely in a YAML file or dict.

    The ``path`` section names the gases and their concentrations, the path
    length, pressure and temperature. Lines of every gas are fetched from
    HITRAN (all isotopologues by default, cached locally), a
    :class:`HomogeneousPath` is built, and :func:`run_forward` is called with
    the rest of the config.

    Gases without lines in the range (or not served line by line by
    HITRANonline) do not absorb there: they are left out with a warning and
    listed in ``result.skipped_gases``.

    Example YAML::

        spectral:  {nu_min_cm1: 600, nu_max_cm1: 2500, step_cm1: 0.01}
        path:
          length_m: 1.0
          pressure_hpa: 1013.0
          temperature_k: 288.2
          gas_units: vmr          # vmr | ppm | atm (partial pressure)
          gases: {H2O: 0.0188, CO2: 3.3e-4, CH4: 1.7e-6}
        output:    {per_gas: true, bin_um: 0.02, attenuation_db_m: true}

    >>> result = run_path("examples/standoff_7gases.yaml")
    >>> result.wavelength_um, result.transmittance_binned
    >>> result.transmittance_per_gas_binned["CO2"]
    """
    from .data_loader import from_hapi
    from .hitran_manager import HITRANNoLinesError, HITRANNotAvailableError

    cfg = load_config(config)
    if cfg["path"] is None:
        raise ValueError("run_path needs a 'path' section (length_m, gases, …) in the config")
    pa = {**_PATH_DEFAULTS, **cfg["path"]}
    sp, ca = cfg["spectral"], cfg["calculation"]
    vmr = _path_vmr(pa)

    # Lines up to the cutoff beyond the range also reach into it
    margin = float(ca["line_cutoff_cm1"])
    lo = max(float(sp["nu_min_cm1"]) - margin, 0.0)
    hi = float(sp["nu_max_cm1"]) + margin
    # One folder per gas: HAPI reloads every table of the folder on each download
    cache = Path(pa["cache_dir"]).expanduser() if pa["cache_dir"] else DEFAULT_CACHE_DIR

    loaded, skipped = [], {}
    for gas in vmr:
        try:
            loaded.append(from_hapi(gas, lo, hi, iso_id=pa["isotopologues"],
                                    cache_dir=cache / gas))
        except (HITRANNoLinesError, HITRANNotAvailableError) as err:
            logger.warning("run_path: %s left out (%s)", gas, err)
            skipped[gas] = f"{type(err).__name__}: {err}"
    if not loaded:
        raise HITRANNoLinesError(
            f"None of the gases {list(vmr)} has lines in {lo:.2f}–{hi:.2f} cm⁻¹: {skipped}")

    path = HomogeneousPath(pa["length_m"], vmr, pressure_hpa=pa["pressure_hpa"],
                           temperature_k=pa["temperature_k"])
    result = run_forward(SpectralData.combine(loaded), path, cfg)
    result.skipped_gases = skipped
    if cfg["output"]["per_gas"]:
        # A gas with no lines does not absorb: T = 1, A = 0
        for gas in skipped:
            ones = np.ones_like(result.wavenumber_cm1)
            result.transmittance_per_gas[gas] = ones
            if result.transmittance_per_gas_binned is not None:
                result.transmittance_per_gas_binned[gas] = np.ones_like(result.wavelength_um)
            if result.attenuation_per_gas_db_m is not None:
                n = (result.wavelength_um if result.wavelength_um is not None
                     else result.wavenumber_cm1).size
                result.attenuation_per_gas_db_m[gas] = np.zeros(n)
    return result


# ── File output ───────────────────────────────────────────────────────────────

def _write(result: ForwardResult, path: str, fmt: str, prefix: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Columns on the wavenumber grid and on the wavelength bins
    grid = {"wavenumber_cm1": result.wavenumber_cm1}
    if result.transmittance is not None:
        grid["transmittance"] = result.transmittance
    if result.optical_depth is not None:
        grid["optical_depth"] = result.optical_depth
    for g, T in (result.transmittance_per_gas or {}).items():
        grid[f"transmittance_{g}"] = T
    binned = {}
    if result.wavelength_um is not None:
        binned = {"wavelength_um": result.wavelength_um,
                  "transmittance": result.transmittance_binned}
        for g, T in (result.transmittance_per_gas_binned or {}).items():
            binned[f"transmittance_{g}"] = T
    att_on = binned if binned else grid
    if result.attenuation_db_m is not None:
        att_on["attenuation_db_m"] = result.attenuation_db_m
    for g, A in (result.attenuation_per_gas_db_m or {}).items():
        att_on[f"attenuation_{g}_db_m"] = A

    if fmt == "npz":
        arrays = dict(grid)
        if result.optical_depth_layers is not None:
            arrays["optical_depth_layers"] = result.optical_depth_layers
        arrays.update({f"binned_{k}": v for k, v in binned.items()})
        out_path = out_dir / f"{prefix}.npz"
        np.savez(out_path, **arrays)

    else:  # csv or txt: the grid in <prefix>.<fmt>, the bins in <prefix>_binned.<fmt>
        sep = "," if fmt == "csv" else "\t"
        out_path = out_dir / f"{prefix}.{fmt}"
        for target, cols in ((out_path, grid), (out_dir / f"{prefix}_binned.{fmt}", binned)):
            if cols:
                np.savetxt(target, np.column_stack(list(cols.values())), delimiter=sep,
                           header=sep.join(cols), comments="")
                print(f"Saved: {target}")
        return out_path

    print(f"Saved: {out_path}")
    return out_path
