"""Explicit, grid-checked inputs for the final 30,000-scene recipe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .forward_adapter import NU_CHANNEL
from .state import build_state_matrix_full, split_indices


def load_final_dataset(
    states_path: str | Path,
    radiances_path: str | Path,
    noise_path: str | Path,
    *,
    split_path: str | Path | None = None,
    tud_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load aligned NWP, 714 cm⁻¹ radiance, noise, angle and optional TUD.

    All artifact paths are explicit. The manifest/download layer validates their
    hashes before this function reads them.
    """
    with np.load(states_path) as archive:
        states = {key: archive[key] for key in archive.files}
    with np.load(radiances_path) as archive:
        radiance = {key: archive[key] for key in archive.files}
    with np.load(noise_path) as archive:
        noise = {key: archive[key] for key in archive.files}
    clean = np.asarray(radiance["L"], dtype=np.float64)
    eps = np.asarray(radiance["eps"], dtype=np.float64)
    realization = np.asarray(noise["noise"], dtype=np.float64)
    instrument_sigma = np.asarray(noise["sigma_ch"], dtype=np.float64)
    mu = np.asarray(radiance["mu"], dtype=np.float32)
    n = 30_000
    channels = len(NU_CHANNEL)
    for name, value, shape in (
        ("clean radiance", clean, (n, channels)),
        ("emissivity", eps, (n, channels)),
        ("noise realization", realization, (n, channels)),
        ("instrument sigma", instrument_sigma, (channels,)),
        ("scan cosine", mu, (n,)),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
    for name, archive in (("radiance", radiance), ("noise", noise)):
        if "nu" in archive and not np.allclose(archive["nu"], NU_CHANNEL, atol=1.0e-9):
            raise ValueError(f"{name} artifact uses a different spectral grid")
    if split_path is None:
        indices = split_indices()
    else:
        with np.load(split_path) as archive:
            indices = {key: np.asarray(archive[key]) for key in ("train", "val", "test")}
        expected = split_indices()
        for key in expected:
            if not np.array_equal(indices[key], expected[key]):
                raise ValueError(
                    f"Materialized {key} split differs from the published seed-42 split"
                )
    result: dict[str, Any] = {
        "d": states,
        "X": build_state_matrix_full(states),
        "Y": clean,
        "Yn": clean + realization,
        "EPS": eps,
        "sigma_inst": instrument_sigma,
        "mu": mu,
        "idx": indices,
        "P_lay": np.asarray(states["P_lay"], dtype=np.float32),
    }
    if result["P_lay"].shape != (n, 136):
        raise ValueError("Pressure layers must have shape (30000, 136)")
    if tud_path is not None:
        tud = np.load(tud_path, mmap_mode="r")
        if tud.shape != (n, 3 * channels):
            raise ValueError("TUD must have shape (30000, 6435) and order [T,U,D]")
        result["TUD"] = tud
    return result


def load_sigma_total(instrument_sigma: np.ndarray, model_error_path: str | Path) -> np.ndarray:
    """Return sqrt(instrument² + forward-error²) per spectral channel."""
    with np.load(model_error_path) as archive:
        model_error = np.asarray(archive["s_f"], dtype=np.float64)
        if "nu" in archive and not np.allclose(archive["nu"], NU_CHANNEL, atol=1.0e-9):
            raise ValueError("Forward-error artifact uses a different spectral grid")
    instrument_sigma = np.asarray(instrument_sigma, dtype=np.float64)
    if instrument_sigma.shape != model_error.shape or model_error.shape != (len(NU_CHANNEL),):
        raise ValueError("Noise and forward-error sigma must have 2145 channels")
    return np.sqrt(instrument_sigma**2 + model_error**2)
