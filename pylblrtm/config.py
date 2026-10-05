"""
config.py — YAML-driven forward model runner.

Separates *what to compute* (the YAML) from *the data* (SpectralData + profile).
The user edits forward_config.yaml and passes it to run_forward(); the spectral
data and atmospheric profile are always provided as Python objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .spectral_data import SpectralData
from .atmosphere import AtmosphericProfile
from .forward import _build_optical_depth


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
    },
    "output": {
        "transmittance": True,
        "optical_depth": False,
        "optical_depth_layers": False,
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

    requested = [out["transmittance"], out["optical_depth"], out["optical_depth_layers"]]
    if not any(requested):
        raise ValueError(
            "At least one output must be enabled: "
            "transmittance, optical_depth, or optical_depth_layers"
        )

    fmt = out["save"]["format"]
    if fmt not in {"npz", "csv", "txt"}:
        raise ValueError(
            f"output.save.format must be 'npz', 'csv', or 'txt'; got '{fmt}'"
        )


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
    config : dict
        The fully resolved configuration used for this run.
    """

    wavenumber_cm1: np.ndarray
    config: dict
    transmittance: Optional[np.ndarray] = field(default=None)
    optical_depth: Optional[np.ndarray] = field(default=None)
    optical_depth_layers: Optional[np.ndarray] = field(default=None)

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

    # Core line-by-line optical depth computation
    od = _build_optical_depth(
        nu,
        spectral_data,
        layers,
        line_cutoff_cm1=float(ca["line_cutoff_cm1"]),
        tile_width_cm1=float(ca["tile_width_cm1"]),
    )

    mu = float(ca["mu_view"])

    # Assemble requested outputs
    result = ForwardResult(wavenumber_cm1=nu, config=cfg)

    if out["transmittance"]:
        result.transmittance = np.exp(-od.sum(axis=0) / mu)

    if out["optical_depth"]:
        result.optical_depth = od.sum(axis=0)

    if out["optical_depth_layers"]:
        result.optical_depth_layers = od

    # Auto-save if the config requests it
    if out["save"]["enabled"]:
        result.save()

    return result


# ── File output ───────────────────────────────────────────────────────────────

def _write(result: ForwardResult, path: str, fmt: str, prefix: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "npz":
        arrays: dict = {"wavenumber_cm1": result.wavenumber_cm1}
        if result.transmittance is not None:
            arrays["transmittance"] = result.transmittance
        if result.optical_depth is not None:
            arrays["optical_depth"] = result.optical_depth
        if result.optical_depth_layers is not None:
            arrays["optical_depth_layers"] = result.optical_depth_layers
        out_path = out_dir / f"{prefix}.npz"
        np.savez(out_path, **arrays)

    else:  # csv or txt
        sep = "," if fmt == "csv" else "\t"
        cols = [result.wavenumber_cm1]
        headers = ["wavenumber_cm1"]
        if result.transmittance is not None:
            cols.append(result.transmittance)
            headers.append("transmittance")
        if result.optical_depth is not None:
            cols.append(result.optical_depth)
            headers.append("optical_depth")
        out_path = out_dir / f"{prefix}.{fmt}"
        np.savetxt(
            out_path,
            np.column_stack(cols),
            delimiter=sep,
            header=sep.join(headers),
            comments="",
        )

    print(f"Saved: {out_path}")
    return out_path
