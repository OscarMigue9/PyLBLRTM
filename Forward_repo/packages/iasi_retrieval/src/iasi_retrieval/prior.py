"""Geographic prior tied to the frozen final APAE decoder.

Source: ``model`` commit 9f3e0a4, ``prior_z.py``. The order of geographic
features is latitude, longitude, day of year, surface pressure in hPa.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .models import APAE
from .state import Standardizer, block_mse, state_metrics_full

FINAL_HIDDEN = (256, 256, 256, 256)


def geographic_features(data: Mapping[str, np.ndarray], doy: np.ndarray) -> np.ndarray:
    """Return the four raw features in their checkpoint order."""
    features = np.column_stack((data["lat"], data["lon"], doy, data["P_surf_hPa"]))
    return features.astype(np.float64)


def make_prior_network(
    input_dim: int = 4, hidden: tuple[int, ...] = FINAL_HIDDEN, latent_dim: int = 24
) -> nn.Sequential:
    """Construct the exact Linear/GELU topology stored in the prior checkpoint."""
    modules: list[nn.Module] = []
    width = input_dim
    for next_width in hidden:
        modules.extend((nn.Linear(width, int(next_width)), nn.GELU()))
        width = int(next_width)
    modules.append(nn.Linear(width, latent_dim))
    return nn.Sequential(*modules)


def load_prior(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Sequential, dict[str, Any]]:
    """Load a prior and validate its topology from the saved tensor shapes.

    The frozen historical artifact only contains ``state_dict``, ``capas``,
    ``mu_p``, ``sd_p`` and ``epocas``. Later checkpoints may also provide
    ``lat_dim`` and ``y_t_dim``; when present, these must agree with the tensors.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        hidden = tuple(int(width) for width in checkpoint["capas"])
        weights = checkpoint["state_dict"]
        feature_mean = np.asarray(checkpoint["mu_p"], dtype=np.float64)
        feature_sd = np.asarray(checkpoint["sd_p"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Prior checkpoint lacks a required architecture or normalization field"
        ) from exc
    if hidden != FINAL_HIDDEN:
        raise ValueError("The final geographic prior must have four 256-unit layers")
    final_index = 2 * len(hidden)
    first_weight = weights.get("0.weight")
    final_weight = weights.get(f"{final_index}.weight")
    final_bias = weights.get(f"{final_index}.bias")
    if (
        not isinstance(first_weight, torch.Tensor)
        or first_weight.shape != (hidden[0], 4)
        or not isinstance(final_weight, torch.Tensor)
        or final_weight.shape != (24, hidden[-1])
        or not isinstance(final_bias, torch.Tensor)
        or final_bias.shape != (24,)
    ):
        raise ValueError("Prior state_dict must map four features through 256x4 to z24")
    if "lat_dim" in checkpoint and int(checkpoint["lat_dim"]) != final_weight.shape[0]:
        raise ValueError("Prior latent metadata disagrees with the saved output layer")
    if "y_t_dim" in checkpoint and int(checkpoint["y_t_dim"]) != 6435:
        raise ValueError("Prior metadata does not match the 2145-channel APAE")
    if feature_mean.shape != (4,) or feature_sd.shape != (4,):
        raise ValueError("Prior feature normalization must contain four values")
    if (
        not np.isfinite(feature_mean).all()
        or not np.isfinite(feature_sd).all()
        or np.any(feature_sd <= 0)
    ):
        raise ValueError("Prior feature normalization must be finite with positive scales")
    network = make_prior_network(hidden=hidden).to(device)
    network.load_state_dict(weights, strict=True)
    network.eval()
    return network, checkpoint


def predict_prior(
    network: nn.Module,
    checkpoint: Mapping[str, Any],
    features: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Predict initial latents from raw geographic features."""
    if features.ndim != 2 or features.shape[1] != 4:
        raise ValueError("Geographic features must have shape (N, 4)")
    normalized = (features - checkpoint["mu_p"]) / checkpoint["sd_p"]
    return network(torch.as_tensor(normalized, dtype=torch.float32, device=device))


def train_prior(
    apae: APAE,
    apae_checkpoint: Mapping[str, Any],
    state: np.ndarray,
    features: np.ndarray,
    *,
    epochs: int = 400,
    batch_size: int = 512,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 40,
    seed: int = 7,
    device: torch.device | str = "cpu",
) -> tuple[nn.Sequential, dict[str, Any]]:
    """Train the final prior through the frozen APAE decoder.

    The objective is seven-block MSE in standardized state space, not a latent
    MSE. Validation and test rows come from the APAE checkpoint itself.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if apae_checkpoint.get("n_aux", 0):
        raise ValueError("The published prior requires an APAE without auxiliary input")
    splits = {key: np.asarray(value) for key, value in apae_checkpoint["splits"].items()}
    train, validation, test = (splits[key] for key in ("train", "val", "test"))
    if state.shape != (len(features), 818) or features.shape != (len(state), 4):
        raise ValueError("Expected matching physical states (N, 818) and features (N, 4)")
    sx = Standardizer.from_state(apae_checkpoint["sx"])
    feature_mean = features[train].mean(axis=0)
    feature_sd = features[train].std(axis=0)
    normalized_features = torch.as_tensor(
        (features - feature_mean) / feature_sd, dtype=torch.float32
    )
    normalized_state = torch.as_tensor(sx.to_z(state), dtype=torch.float32)
    model = make_prior_network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def epoch(rows: np.ndarray, train_mode: bool) -> float:
        model.train(train_mode)
        order = torch.randperm(len(rows)) if train_mode else torch.arange(len(rows))
        total = 0.0
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[order[start : start + batch_size].numpy()]
            batch_features = normalized_features[batch_rows].to(device)
            batch_state = normalized_state[batch_rows].to(device)
            with torch.set_grad_enabled(train_mode):
                decoded, _ = apae.decode(model(batch_features))
                loss = block_mse(decoded, batch_state)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * len(batch_rows)
        return total / len(rows)

    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    history = []
    for number in range(1, epochs + 1):
        train_loss = epoch(train, True)
        val_loss = epoch(validation, False)
        history.append({"epoch": number, "train": train_loss, "val": val_loss})
        if val_loss < best - 1.0e-7:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        decoded, _ = apae.decode(model(normalized_features[test].to(device)))
    estimate = sx.to_phys(decoded.cpu().numpy().astype(np.float64))
    metrics = state_metrics_full(estimate, state[test])
    checkpoint = {
        "state_dict": best_state,
        "capas": list(FINAL_HIDDEN),
        "mu_p": feature_mean,
        "sd_p": feature_sd,
        "lat_dim": 24,
        "y_t_dim": apae_checkpoint["y_t_dim"],
        "best_val": best,
        "test": metrics,
        "history": history,
        "seed": seed,
    }
    return model, checkpoint
