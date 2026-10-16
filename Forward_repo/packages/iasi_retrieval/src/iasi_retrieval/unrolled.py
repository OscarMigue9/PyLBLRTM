"""Shared-step residual retrieval for the published ``unroll714_sew`` model.

Source: ``model`` commit 9f3e0a4, ``exp18_unrolled.py`` and
``exp16_gn_refino.py``. T=3 was trained; T=4 is the published evaluation
choice. Residual order is the full 2145-channel grid with uncovered channels
set to zero before and after ISRF whitening.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .forward_adapter import NU0, NU1, NU_CHANNEL, SIGMA_CHANNEL, WINDOW_MARGIN, WINDOW_WIDTH
from .models import APAE, ResidualStepNetwork
from .state import Standardizer, block_mse, state_metrics_full


SEW_INPUT_IDENTIFIERS = frozenset(
    {
        "configuration",
        "states_30k",
        "dates_30k",
        "radiances_iasiang",
        "noise_iasiang",
        "split_30k_seed42",
        "model_error_sf",
        "r_self",
        "lut_h2o",
        "lut_co2",
        "lut_o3",
        "lut_n2o",
        "lut_ch4",
        "apae",
        "prior",
        "prior_latents",
    }
)


class WindowForward(Protocol):
    """The small forward interface required by the shared-step model."""

    def _win(self, lo: float) -> tuple: ...

    def window(
        self, state: torch.Tensor, rows: np.ndarray, lo: float, mu: torch.Tensor | None = None
    ) -> tuple[torch.Tensor | None, np.ndarray]: ...


def full_band_windows() -> list[float]:
    """Return the exact 30 overlapping 25 cm⁻¹ windows of the final recipe."""
    step = WINDOW_WIDTH - 2 * WINDOW_MARGIN - 1.0
    windows = []
    lo = NU0
    while lo < NU1 - WINDOW_MARGIN:
        windows.append(float(min(lo, NU1 - WINDOW_WIDTH)))
        lo += step
    return windows


@dataclass(frozen=True)
class WindowSelection:
    lo: float
    keep: np.ndarray
    absolute: np.ndarray


def prepare_windows(
    forward: WindowForward, sigma: np.ndarray, windows: list[float] | None = None
) -> list[WindowSelection]:
    """Assign each finite-noise channel to its first covering window."""
    sigma = np.asarray(sigma)
    if sigma.shape != (len(NU_CHANNEL),):
        raise ValueError("Sigma must have one value per 2145-channel grid point")
    seen = ~np.isfinite(sigma)
    selected = []
    for lo in full_band_windows() if windows is None else windows:
        mask = forward._win(lo)[2]
        absolute = np.flatnonzero(mask)
        keep = ~seen[absolute]
        if keep.sum() < 4:
            continue
        seen[absolute[keep]] = True
        selected.append(WindowSelection(lo, keep, absolute[keep]))
    if not selected:
        raise ValueError("No forward window covers enough finite-noise channels")
    all_channels = np.concatenate([window.absolute for window in selected])
    if len(np.unique(all_channels)) != len(all_channels):
        raise ValueError("Overlapping forward windows assigned a channel twice")
    return selected


def whitening_matrix(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    channels: np.ndarray = NU_CHANNEL,
) -> torch.Tensor:
    """Construct the historical inverse ISRF Cholesky factor."""
    frequencies = torch.as_tensor(channels, dtype=torch.float64, device=device)
    separation = frequencies[:, None] - frequencies[None, :]
    correlation = torch.exp(-(separation**2) / (4.0 * SIGMA_CHANNEL**2))
    correlation = torch.where(
        separation.abs() > 6.0 * SIGMA_CHANNEL, torch.zeros_like(correlation), correlation
    )
    correlation = 0.5 * (correlation + correlation.T)
    correlation += 1.0e-9 * torch.eye(len(frequencies), dtype=torch.float64, device=device)
    lower = torch.linalg.cholesky(correlation)
    inverse = torch.linalg.solve_triangular(
        lower, torch.eye(len(frequencies), dtype=torch.float64, device=device), upper=False
    )
    return inverse.to(dtype)


def auxiliary_features(
    pressure_surface: np.ndarray, emissivity: np.ndarray, mu: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Fit and apply [surface pressure, scan cosine, emissivity] scaling."""
    pressure_surface = np.asarray(pressure_surface, dtype=np.float64)
    emissivity = np.asarray(emissivity, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    train_rows = np.asarray(train_rows)
    if (
        emissivity.shape != (len(pressure_surface), len(NU_CHANNEL))
        or mu.shape != pressure_surface.shape
    ):
        raise ValueError("Expected pressure (N,), scan cosine (N,) and emissivity (N, 2145)")
    pressure_mean = float(pressure_surface[train_rows].mean())
    pressure_sd = float(pressure_surface[train_rows].std())
    mu_mean = float(mu[train_rows].mean())
    mu_sd = float(mu[train_rows].std()) + 1.0e-9
    emissivity_mean = emissivity[train_rows].mean(axis=0)
    emissivity_sd = emissivity[train_rows].std(axis=0) + 1.0e-9
    stats = {
        "pressure_mean": pressure_mean,
        "pressure_sd": pressure_sd,
        "mu_mean": mu_mean,
        "mu_sd": mu_sd,
        "emissivity_mean": emissivity_mean,
        "emissivity_sd": emissivity_sd,
    }
    return apply_auxiliary_features(pressure_surface, emissivity, mu, stats), stats


def apply_auxiliary_features(
    pressure_surface: np.ndarray,
    emissivity: np.ndarray,
    mu: np.ndarray,
    stats: Mapping[str, np.ndarray | float],
) -> np.ndarray:
    """Apply the frozen train statistics in [surface pressure, mu, EPS] order."""
    pressure_surface = np.asarray(pressure_surface, dtype=np.float64)
    emissivity = np.asarray(emissivity, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if (
        pressure_surface.ndim != 1
        or mu.shape != pressure_surface.shape
        or emissivity.shape != (len(pressure_surface), len(NU_CHANNEL))
    ):
        raise ValueError("Auxiliary inputs must have shapes (N,), (N,), (N, 2145)")
    if (
        not np.isfinite(pressure_surface).all()
        or not np.isfinite(mu).all()
        or not np.isfinite(emissivity).all()
    ):
        raise ValueError("Auxiliary inputs must be finite")
    features = np.concatenate(
        (
            ((pressure_surface - stats["pressure_mean"]) / stats["pressure_sd"])[:, None],
            ((mu - stats["mu_mean"]) / stats["mu_sd"])[:, None],
            (emissivity - stats["emissivity_mean"]) / stats["emissivity_sd"],
        ),
        axis=1,
    ).astype(np.float32)
    return features


def residual(
    forward: WindowForward,
    state: torch.Tensor,
    rows: np.ndarray,
    windows: list[WindowSelection],
    observed: torch.Tensor,
    sigma: torch.Tensor,
    valid: torch.Tensor,
    whitening: torch.Tensor,
    mu: torch.Tensor | None = None,
    checkpoint_forward: bool = False,
) -> torch.Tensor:
    """Compute the differentiable, fully whitened residual on the fixed grid."""
    result = torch.zeros(len(state), len(NU_CHANNEL), device=state.device, dtype=state.dtype)
    for selected in windows:

        def evaluate(x: torch.Tensor, lo: float = selected.lo) -> torch.Tensor:
            radiance, _ = forward.window(x, rows, lo, mu=mu)
            if radiance is None:
                raise RuntimeError(f"Window {lo} no longer covers its selected channels")
            return radiance

        use_checkpoint = checkpoint_forward and torch.is_grad_enabled()
        radiance = (
            activation_checkpoint(evaluate, state, use_reentrant=False)
            if use_checkpoint
            else evaluate(state)
        )
        absolute = selected.absolute
        result[:, absolute] = (observed[:, absolute] - radiance[:, selected.keep]) / sigma[absolute]
    result = result * valid
    return (result @ whitening.T) * valid


class UnrolledRetrieval(nn.Module):
    """Dataset-aligned T-step inference with one shared R2A update network."""

    def __init__(
        self,
        apae: APAE,
        step: ResidualStepNetwork,
        forward: WindowForward,
        standardizer: Standardizer,
        observed: torch.Tensor,
        sigma: torch.Tensor,
        auxiliary: torch.Tensor,
        initial_latents: torch.Tensor,
        mu: torch.Tensor,
        residual_sd: torch.Tensor,
        whitening: torch.Tensor,
        windows: list[WindowSelection],
        clamp: float = 6.0,
    ) -> None:
        super().__init__()
        if step.backbone[0].in_features != 4292 or step.to_mu.out_features != 24:
            raise ValueError("Final step network must map 4292 inputs to z24")
        if observed.shape != (len(initial_latents), len(NU_CHANNEL)):
            raise ValueError("Observed radiance must have shape (N, 2145)")
        if auxiliary.shape != (len(initial_latents), 2147):
            raise ValueError("Auxiliary features must be [pressure, mu, eps(2145)]")
        if initial_latents.shape[1] != 24 or mu.shape != (len(initial_latents),):
            raise ValueError("Expected prior latents (N, 24) and scan cosine (N,)")
        device = observed.device
        self.apae = apae
        self.step = step
        self.forward = forward
        self.standardizer = standardizer
        self.observed = observed
        self.sigma = sigma
        self.auxiliary = auxiliary
        self.initial_latents = initial_latents
        self.mu = mu
        self.residual_sd = residual_sd
        self.whitening = whitening
        self.windows = windows
        self.clamp = clamp
        self.valid = torch.as_tensor(
            np.isfinite(sigma.detach().cpu().numpy()), dtype=torch.float32, device=device
        )
        self.state_mean = torch.as_tensor(standardizer.mu, dtype=torch.float32, device=device)
        self.state_sd = torch.as_tensor(standardizer.sd, dtype=torch.float32, device=device)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode and clamp the standardized state before physical scaling."""
        normalized, _ = self.apae.decode(z)
        if self.clamp > 0:
            normalized = torch.clamp(normalized, -self.clamp, self.clamp)
        return normalized * self.state_sd + self.state_mean, normalized

    def run(
        self, rows: np.ndarray, iterations: int = 3, checkpoint_forward: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Return physical state, normalized state, final latent and updates."""
        rows = np.asarray(rows)
        z = self.initial_latents[rows]
        updates = []
        for _ in range(iterations):
            state, _ = self.decode(z)
            whitened = (
                residual(
                    self.forward,
                    state,
                    rows,
                    self.windows,
                    self.observed[rows],
                    self.sigma,
                    self.valid,
                    self.whitening,
                    self.mu[rows],
                    checkpoint_forward=checkpoint_forward,
                )
                / self.residual_sd
            )
            delta, _ = self.step(torch.cat((whitened, self.auxiliary[rows]), dim=1))
            updates.append(delta)
            z = torch.clamp(z + delta, -self.clamp, self.clamp)
        state, normalized = self.decode(z)
        return state, normalized, z, updates

    def evaluate(
        self, rows: np.ndarray, truth: np.ndarray, iterations: int = 4, batch_size: int = 16
    ) -> dict[str, float]:
        """Evaluate the published state metrics on explicit row IDs."""
        outputs = []
        self.step.eval()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                state, _, _, _ = self.run(rows[start : start + batch_size], iterations)
                outputs.append(state.cpu().numpy().astype(np.float64))
        metrics = state_metrics_full(np.concatenate(outputs), truth[rows])
        metrics["gases_dex"] = float(
            np.mean([metrics[f"rmse_{gas}_dex"] for gas in ("H2O", "CO2", "O3", "N2O", "CH4")])
        )
        return metrics


def load_sew(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[ResidualStepNetwork, dict[str, Any]]:
    """Load only a checkpoint compatible with the final ``sew`` recipe."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    arguments = saved.get("args", {})
    if saved.get("latent") != 24 or saved.get("in_dim") != 4292:
        raise ValueError("Final sew checkpoint requires z24 and 4292 step inputs")
    for key, expected in (
        ("se_corr", "isrf"),
        ("angulo", True),
        ("za", "clim"),
        ("labels", "perfect"),
    ):
        if arguments.get(key) != expected:
            raise ValueError(f"Checkpoint {key} must be {expected!r}")
    if float(arguments.get("lam_chi2", float("nan"))) != 0.0:
        raise ValueError("Final sew checkpoint requires zero explicit chi2 weight")
    if np.asarray(saved["R_SD"]).shape != (len(NU_CHANNEL),):
        raise ValueError("Residual scale must contain 2145 channels")
    network = ResidualStepNetwork(4292, 24).to(device)
    network.load_state_dict(saved["mlp"], strict=True)
    network.eval()
    return network, saved


def calibrate_residual_sd(model: UnrolledRetrieval, train_rows: np.ndarray) -> torch.Tensor:
    """Measure channel scales at the geographic prior on first 128 train rows."""
    rows = np.asarray(train_rows)[:128]
    with torch.no_grad():
        state, _ = model.decode(model.initial_latents[rows])
        reference = residual(
            model.forward,
            state,
            rows,
            model.windows,
            model.observed[rows],
            model.sigma,
            model.valid,
            model.whitening,
            model.mu[rows],
        )
    return reference.std(dim=0).clamp_min(1.0e-3)


def _save_checkpoint_atomic(checkpoint: dict[str, Any], path: str | Path) -> None:
    """Replace a checkpoint only after the new file is fully written.

    A Modal Volume may commit while training is still running. Writing to a
    temporary sibling keeps a previously complete checkpoint available while
    the next one is serialized.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(checkpoint, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _validated_input_hashes(hashes: Mapping[str, str] | None) -> dict[str, str]:
    """Require the complete final training input set with canonical SHA256 digests."""
    if hashes is None or set(hashes) != SEW_INPUT_IDENTIFIERS:
        raise ValueError("Sew checkpoints require the complete 16-file input SHA256 manifest")
    result = dict(sorted(hashes.items()))
    for identifier, digest in result.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid SHA256 for sew input {identifier}")
    return result


def _resume_training_state(
    path: str | Path,
    model: UnrolledRetrieval,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    rng: np.random.Generator,
    *,
    steps: int,
    batch_size: int,
    supervised_rows: int,
    iterations: int,
    lr: float,
    seed: int,
    trace_every: int,
    trace_rows: int,
    input_sha256: Mapping[str, str],
    legacy_checkpoint_sha256: str | None,
) -> tuple[int, list[dict[str, float]], np.ndarray, np.ndarray, bool]:
    """Restore the next optimizer step and every stochastic state in a sew run."""
    if legacy_checkpoint_sha256 is not None:
        legacy_checkpoint_sha256 = legacy_checkpoint_sha256.lower()
        if len(legacy_checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in legacy_checkpoint_sha256
        ):
            raise ValueError("Legacy resume requires a valid checkpoint SHA256")
        checkpoint_bytes = Path(path).read_bytes()
        if hashlib.sha256(checkpoint_bytes).hexdigest() != legacy_checkpoint_sha256:
            raise ValueError("Legacy sew checkpoint SHA256 does not match the approved digest")
        saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    else:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    legacy_resume = "input_sha256" not in saved
    if legacy_resume:
        if legacy_checkpoint_sha256 is None:
            raise ValueError(
                "Legacy sew checkpoint lacks input hashes; supply its independently verified SHA256"
            )
    else:
        if legacy_checkpoint_sha256 is not None:
            raise ValueError("Legacy SHA256 exception is only for checkpoints without input hashes")
        if _validated_input_hashes(saved["input_sha256"]) != input_sha256:
            raise ValueError("Resume checkpoint input SHA256 manifest differs from current inputs")
    arguments = saved.get("args", {})
    expected_args = {
        "se_corr": "isrf",
        "angulo": True,
        "za": "clim",
        "labels": "perfect",
        "lam_chi2": 0.0,
        "iters": iterations,
        "seed": seed,
    }
    if any(arguments.get(key) != value for key, value in expected_args.items()):
        raise ValueError("Resume checkpoint has a different sew recipe or seed")
    if saved.get("latent") != 24 or saved.get("in_dim") != 4292:
        raise ValueError("Resume checkpoint has a different network architecture")
    recorded_training = saved.get("training")
    if recorded_training is not None:
        expected_training = {
            "steps": steps,
            "batch_size": batch_size,
            "supervised_rows": supervised_rows,
            "iterations": iterations,
            "lr": lr,
            "trace_every": trace_every,
            "trace_rows": trace_rows,
            "device_type": model.observed.device.type,
        }
        if any(recorded_training.get(key) != value for key, value in expected_training.items()):
            raise ValueError("Resume checkpoint has different training settings")
    step_number = saved.get("paso")
    if not isinstance(step_number, int) or not 0 < step_number <= steps:
        raise ValueError("Resume checkpoint has an invalid completed step")
    curve = np.asarray(saved["curve"], dtype=np.float32)
    update_norms = np.asarray(saved["update_norms"], dtype=np.float32)
    if curve.shape != (steps,) or update_norms.shape != (steps, iterations):
        raise ValueError("Resume checkpoint has incompatible training arrays")
    if not np.isfinite(curve[:step_number]).all() or np.isfinite(curve[step_number:]).any():
        raise ValueError("Resume checkpoint loss curve is incomplete or overfilled")
    if saved["scheduler"].get("last_epoch") != step_number:
        raise ValueError("Resume checkpoint scheduler does not match the completed step")
    residual_sd = torch.as_tensor(saved["R_SD"], device=model.residual_sd.device)
    if not torch.equal(residual_sd, model.residual_sd):
        raise ValueError("Resume checkpoint uses a different calibrated residual scale")
    cuda_rng = saved["cuda_rng"]
    if (model.observed.device.type == "cuda") != bool(cuda_rng):
        raise ValueError("Resume requires the same CPU or CUDA backend")
    if cuda_rng and len(cuda_rng) != torch.cuda.device_count():
        raise ValueError("Resume requires the same number of visible CUDA devices")
    model.step.load_state_dict(saved["mlp"], strict=True)
    optimizer.load_state_dict(saved["opt"])
    scheduler.load_state_dict(saved["scheduler"])
    rng.bit_generator.state = saved["numpy_rng"]
    torch.set_rng_state(saved["torch_rng"])
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    return step_number, saved["trace"], curve, update_norms, legacy_resume


def train_unrolled(
    model: UnrolledRetrieval,
    truth: np.ndarray,
    splits: Mapping[str, np.ndarray],
    *,
    steps: int = 1000,
    batch_size: int = 16,
    supervised_rows: int = 24_000,
    iterations: int = 3,
    lr: float = 1.0e-3,
    seed: int = 0,
    trace_every: int = 50,
    trace_rows: int = 256,
    checkpoint_every: int = 100,
    checkpoint_path: str | Path | None = None,
    resume_path: str | Path | None = None,
    input_sha256: Mapping[str, str] | None = None,
    legacy_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Train the shared update with the historical supervised objective.

    The forward is evaluated at every step and remains in the backward graph.
    Checkpoints bind the verified inputs and store RNG/scheduler state. A
    pre-manifest checkpoint requires its independently verified file digest.
    """
    if legacy_checkpoint_sha256 is not None and resume_path is None:
        raise ValueError("Legacy checkpoint SHA256 requires resume_path")
    verified_inputs = (
        _validated_input_hashes(input_sha256)
        if checkpoint_path is not None or resume_path is not None
        else None
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if iterations != 3:
        raise ValueError("The final sew update was trained with exactly three steps")
    train_rows = np.asarray(splits["train"])
    validation = np.asarray(splits["val"])[:trace_rows]
    pool = train_rows[:supervised_rows]
    target = torch.as_tensor(
        model.standardizer.to_z(truth), dtype=torch.float32, device=model.observed.device
    )
    rng = np.random.default_rng(seed + 7)
    optimizer = torch.optim.AdamW(model.step.parameters(), lr=lr, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    history = []
    curve = np.full(steps, np.nan, dtype=np.float32)
    update_norms = np.full((steps, iterations), np.nan, dtype=np.float32)
    first_step = 0
    legacy_resume = False
    if resume_path is not None:
        first_step, history, curve, update_norms, legacy_resume = _resume_training_state(
            resume_path,
            model,
            optimizer,
            scheduler,
            rng,
            steps=steps,
            batch_size=batch_size,
            supervised_rows=supervised_rows,
            iterations=iterations,
            lr=lr,
            seed=seed,
            trace_every=trace_every,
            trace_rows=trace_rows,
            input_sha256=verified_inputs,
            legacy_checkpoint_sha256=legacy_checkpoint_sha256,
        )
    for step_number in range(first_step, steps):
        if trace_every > 0 and step_number % trace_every == 0:
            history.append(
                {
                    "step": step_number,
                    **model.evaluate(
                        validation, truth, iterations=iterations, batch_size=batch_size
                    ),
                }
            )
        rows = rng.choice(pool, min(batch_size, len(pool)), replace=False)
        model.step.train()
        optimizer.zero_grad(set_to_none=True)
        _, normalized, _, deltas = model.run(rows, iterations, checkpoint_forward=True)
        loss = block_mse(normalized, target[rows])
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite supervised loss at step {step_number}")
        curve[step_number] = float(loss.detach())
        for index, delta in enumerate(deltas):
            update_norms[step_number, index] = float(delta.detach().norm(dim=1).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.step.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if checkpoint_path is not None and (
            step_number + 1 == steps
            or (checkpoint_every > 0 and (step_number + 1) % checkpoint_every == 0)
        ):
            _save_checkpoint_atomic(
                {
                    "mlp": model.step.state_dict(),
                    "opt": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "paso": step_number + 1,
                    "args": {
                        "se_corr": "isrf",
                        "angulo": True,
                        "za": "clim",
                        "labels": "perfect",
                        "lam_chi2": 0.0,
                        "iters": 3,
                        "seed": seed,
                    },
                    "in_dim": 4292,
                    "latent": 24,
                    "R_SD": model.residual_sd.detach().cpu(),
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "input_sha256": verified_inputs,
                    "training": {
                        "steps": steps,
                        "batch_size": batch_size,
                        "supervised_rows": supervised_rows,
                        "iterations": iterations,
                        "lr": lr,
                        "trace_every": trace_every,
                        "trace_rows": trace_rows,
                        "device_type": model.observed.device.type,
                    },
                    "trace": history,
                    "curve": curve,
                    "update_norms": update_norms,
                },
                checkpoint_path,
            )
    return {
        "trace": history,
        "curve": curve,
        "update_norms": update_norms,
        "steps": steps,
        "seed": seed,
        "resumed_from_step": first_step,
        "legacy_resume_without_embedded_input_hashes": legacy_resume,
    }
