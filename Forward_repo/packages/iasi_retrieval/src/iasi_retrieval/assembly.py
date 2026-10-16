"""Wire verified final artifacts into one inference object."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data import load_sigma_total
from .forward_adapter import PhysicalForward
from .mar64 import Mar64Scenes
from .models import ResidualStepNetwork, load_apae
from .prior import geographic_features, load_prior, predict_prior
from .state import Standardizer, split_indices
from .training import set_seed
from .unrolled import (
    UnrolledRetrieval,
    apply_auxiliary_features,
    auxiliary_features,
    calibrate_residual_sd,
    load_sew,
    prepare_windows,
    whitening_matrix,
)


def load_final_retrieval(
    dataset: dict,
    dates_path: str | Path,
    apae_path: str | Path,
    prior_path: str | Path,
    sew_path: str | Path,
    model_error_path: str | Path,
    lut_dir: str | Path,
    r_self_path: str | Path,
    *,
    device: torch.device | str = "cuda",
) -> UnrolledRetrieval:
    """Load the full z24/Ncol, 714–1250 cm⁻¹ ``sew`` inference chain.

    ``dataset`` comes from :func:`iasi_retrieval.data.load_final_dataset`.
    Artifact hash checking belongs to the manifest layer before this call.
    """
    return _assemble(
        dataset,
        dates_path,
        apae_path,
        prior_path,
        model_error_path,
        lut_dir,
        r_self_path,
        sew_path=sew_path,
        device=device,
    )


def bind_mar64(
    base: UnrolledRetrieval, dataset: dict, scenes: Mar64Scenes, prior_path: str | Path
) -> UnrolledRetrieval:
    """Apply the trained retrieval to sea scenes without duplicating the LUT.

    The simulated training rows determine the auxiliary normalizers and water
    climatology. The bound forward has independent 64-row scene tables; its LUT
    tensors remain shared with ``base`` and are never changed here.
    """
    if scenes.truth.shape != (64, 818) or scenes.geography.shape != (64, 4):
        raise ValueError("The frozen sea set must contain 64 full states and geography rows")
    device = base.observed.device
    _, stats = auxiliary_features(
        dataset["d"]["P_surf_hPa"],
        dataset["EPS"],
        dataset["mu"],
        np.asarray(dataset["idx"]["train"]),
    )
    features = apply_auxiliary_features(
        scenes.pressure_surface, scenes.emissivity, scenes.mu, stats
    )
    prior, checkpoint = load_prior(prior_path, device)
    with torch.no_grad():
        initial_latents = predict_prior(prior, checkpoint, scenes.geography, device=device).detach()
    sea_forward = base.forward.with_scene_tables(scenes.pressure_layers, scenes.emissivity)
    return UnrolledRetrieval(
        apae=base.apae,
        step=base.step,
        forward=sea_forward,
        standardizer=base.standardizer,
        observed=torch.as_tensor(scenes.observed, dtype=torch.float32, device=device),
        sigma=base.sigma,
        auxiliary=torch.as_tensor(features, dtype=torch.float32, device=device),
        initial_latents=initial_latents,
        mu=torch.as_tensor(scenes.mu, dtype=torch.float32, device=device),
        residual_sd=base.residual_sd,
        whitening=base.whitening,
        windows=base.windows,
        clamp=base.clamp,
    )


def build_training_retrieval(
    dataset: dict,
    dates_path: str | Path,
    apae_path: str | Path,
    prior_path: str | Path,
    model_error_path: str | Path,
    lut_dir: str | Path,
    r_self_path: str | Path,
    *,
    prior_latents_path: str | Path | None = None,
    seed: int = 0,
    device: torch.device | str = "cuda",
) -> UnrolledRetrieval:
    """Assemble training with the historical precomputed ``za`` when supplied."""
    return _assemble(
        dataset,
        dates_path,
        apae_path,
        prior_path,
        model_error_path,
        lut_dir,
        r_self_path,
        prior_latents_path=prior_latents_path,
        seed=seed,
        device=device,
    )


def _initial_latents(
    prior: torch.nn.Module,
    prior_checkpoint: dict,
    states: dict,
    doy: np.ndarray,
    prior_latents_path: str | Path | None,
    device: torch.device,
) -> torch.Tensor:
    """Use saved training latents or evaluate the frozen prior for inference."""
    if prior_latents_path is not None:
        with np.load(prior_latents_path, allow_pickle=False) as archive:
            if "za" not in archive:
                raise ValueError("Precomputed prior artifact lacks the 'za' array")
            latent = np.asarray(archive["za"])
        if latent.shape != (30_000, 24):
            raise ValueError(
                f"Precomputed prior latents must have shape (30000, 24), got {latent.shape}"
            )
        if not np.issubdtype(latent.dtype, np.number) or not np.isfinite(latent).all():
            raise ValueError("Precomputed prior latents must be finite numeric values")
        return torch.tensor(latent, dtype=torch.float32, device=device)
    geography = geographic_features(states, doy)
    with torch.no_grad():
        return predict_prior(prior, prior_checkpoint, geography, device=device)


def _assemble(
    dataset: dict,
    dates_path: str | Path,
    apae_path: str | Path,
    prior_path: str | Path,
    model_error_path: str | Path,
    lut_dir: str | Path,
    r_self_path: str | Path,
    *,
    sew_path: str | Path | None = None,
    prior_latents_path: str | Path | None = None,
    seed: int | None = None,
    device: torch.device | str,
) -> UnrolledRetrieval:
    device = torch.device(device)
    if seed is not None:
        set_seed(seed)
    apae, apae_checkpoint = load_apae(apae_path, device)
    # The historical trainer initialized APAE before the step network. Keep
    # that RNG order when constructing new weights; the prior was a saved NPZ.
    if sew_path is None:
        step = ResidualStepNetwork(4292, 24).to(device)
        torch.nn.init.zeros_(step.to_mu.weight)
        torch.nn.init.zeros_(step.to_mu.bias)
        sew_checkpoint = None
    else:
        step, sew_checkpoint = load_sew(sew_path, device)
    prior, prior_checkpoint = load_prior(prior_path, device)
    expected = split_indices()
    for key in expected:
        if not np.array_equal(np.asarray(apae_checkpoint["splits"][key]), expected[key]):
            raise ValueError(f"APAE {key} rows differ from the materialized split")
        if not np.array_equal(np.asarray(dataset["idx"][key]), expected[key]):
            raise ValueError(f"Dataset {key} rows differ from the APAE split")
    if "y_t_dim" in prior_checkpoint and prior_checkpoint["y_t_dim"] != apae_checkpoint["y_t_dim"]:
        raise ValueError("Prior and APAE have incompatible TUD grids")
    if prior[-1].out_features != apae_checkpoint["latent_dim"]:
        raise ValueError("Prior and APAE have incompatible latent dimensions")
    # The frozen prior has no APAE identifier or TUD metadata. Its pairing with
    # the decoder must be established by the verified artifact manifest.
    with np.load(dates_path) as dates:
        doy = np.asarray(dates["doy"])
        if "subset_id" in dates and "subset_id" in dataset["d"]:
            if not np.array_equal(dates["subset_id"], dataset["d"]["subset_id"]):
                raise ValueError("Dates and atmospheric states are not row-aligned")
    if doy.shape != (len(dataset["X"]),):
        raise ValueError("Day-of-year vector must have one entry per scene")
    initial_latents = _initial_latents(
        prior, prior_checkpoint, dataset["d"], doy, prior_latents_path, device
    )
    scan_cosine = np.asarray(dataset["mu"], dtype=np.float32)
    if not np.isfinite(scan_cosine).all() or np.any((scan_cosine <= 0) | (scan_cosine > 1)):
        raise ValueError("Final scan cosines must be finite and in (0, 1]")
    features, _ = auxiliary_features(
        dataset["d"]["P_surf_hPa"], dataset["EPS"], scan_cosine, expected["train"]
    )
    sigma = load_sigma_total(dataset["sigma_inst"], model_error_path)
    forward = PhysicalForward.from_dataset(dataset, lut_dir, r_self_path, device)
    windows = prepare_windows(forward, sigma)
    retrieval = UnrolledRetrieval(
        apae=apae,
        step=step,
        forward=forward,
        standardizer=Standardizer.from_state(apae_checkpoint["sx"]),
        observed=torch.as_tensor(dataset["Yn"], dtype=torch.float32, device=device),
        sigma=torch.as_tensor(sigma, dtype=torch.float32, device=device),
        auxiliary=torch.as_tensor(features, dtype=torch.float32, device=device),
        initial_latents=initial_latents,
        mu=torch.as_tensor(scan_cosine, dtype=torch.float32, device=device),
        residual_sd=(
            torch.as_tensor(sew_checkpoint["R_SD"], dtype=torch.float32, device=device)
            if sew_checkpoint is not None
            else torch.ones(2145, dtype=torch.float32, device=device)
        ),
        whitening=whitening_matrix(device),
        windows=windows,
        clamp=float(
            sew_checkpoint.get("args", {}).get("clamp_z", 6.0)
            if sew_checkpoint is not None
            else 6.0
        ),
    )
    if sew_checkpoint is None:
        retrieval.residual_sd = calibrate_residual_sd(retrieval, expected["train"])
    return retrieval
