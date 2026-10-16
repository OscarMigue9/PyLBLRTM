"""APAE training for the final dense 818/6435/z24 configuration.

Source: ``model`` commit 9f3e0a4, ``train_apae.py``. The saved checkpoint
retains the historical keys so the published decoder loader can read it.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import APAE, kl_loss
from .state import Standardizer, block_mse, split_indices, state_metrics_full


def set_seed(seed: int) -> None:
    """Set the same three RNGs used by the historical training scripts."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_apae(
    state: np.ndarray,
    tud: np.ndarray,
    *,
    splits: Mapping[str, np.ndarray] | None = None,
    epochs: int = 300,
    batch_size: int = 512,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    beta: float = 1.0e-4,
    patience: int = 40,
    seed: int = 7,
    device: torch.device | str = "cpu",
) -> tuple[APAE, dict[str, Any]]:
    """Fit APAE on train rows and select weights by validation loss.

    ``tud`` must concatenate transmission, upwelling and downwelling blocks in
    that order. ``beta`` is explicit because the exact published APAE invocation
    must be established from its artifact manifest.
    """
    state = np.asarray(state, dtype=np.float64)
    tud = np.asarray(tud, dtype=np.float64)
    if state.shape != (30_000, 818) or tud.shape != (30_000, 6435):
        raise ValueError("Expected aligned state (30000, 818) and TUD (30000, 6435)")
    set_seed(seed)
    rows = dict(split_indices() if splits is None else splits)
    train, validation, test = (np.asarray(rows[key]) for key in ("train", "val", "test"))
    sx = Standardizer().fit(state[train])
    st = Standardizer().fit(tud[train])
    state_z = torch.as_tensor(sx.to_z(state), dtype=torch.float32)
    tud_z = torch.as_tensor(st.to_z(tud), dtype=torch.float32)

    def batches(selected: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(state_z[selected], tud_z[selected])
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    train_loader = batches(train, True)
    val_loader = batches(validation, False)
    model = APAE(818, 6435, 24, n_aux=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        totals = {key: 0.0 for key in ("loss", "l_a", "l_t", "kl")}
        count = 0
        for atmosphere, transfer in loader:
            atmosphere = atmosphere.to(device)
            transfer = transfer.to(device)
            with torch.set_grad_enabled(train_mode):
                decoded_atmosphere, decoded_transfer, mu, sigma = model(atmosphere, transfer)
                atmosphere_loss = block_mse(decoded_atmosphere, atmosphere)
                transfer_loss = torch.mean((decoded_transfer - transfer) ** 2)
                kl = kl_loss(mu, sigma)
                loss = atmosphere_loss + transfer_loss + beta * kl
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for key, value in (
                ("loss", loss),
                ("l_a", atmosphere_loss),
                ("l_t", transfer_loss),
                ("kl", kl),
            ):
                totals[key] += float(value.detach()) * len(atmosphere)
            count += len(atmosphere)
        return {key: value / count for key, value in totals.items()}

    history = []
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    for number in range(1, epochs + 1):
        train_metrics = epoch(train_loader, True)
        val_metrics = epoch(val_loader, False)
        history.append(
            {
                "epoch": number,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
        )
        if val_metrics["loss"] < best - 1.0e-6:
            best, best_state, stale = val_metrics["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    test_metrics = epoch(batches(test, False), False)
    with torch.no_grad():
        decoded, _, _, _ = model(state_z[test].to(device), tud_z[test].to(device))
    physical = sx.to_phys(decoded.cpu().numpy().astype(np.float64))
    checkpoint = {
        "model_state": best_state,
        "y_a_dim": 818,
        "y_t_dim": 6435,
        "latent_dim": 24,
        "n_aux": 0,
        "sx": sx.state_dict(),
        "st": st.state_dict(),
        "psurf_meta": None,
        "splits": {key: np.asarray(value, dtype=np.int32) for key, value in rows.items()},
        "state": "full",
        "gas": None,
        "no_tud": False,
        "args": {
            "latent_dim": 24,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "beta": beta,
            "patience": patience,
            "seed": seed,
            "state": "full",
            "decoder": "denso",
        },
        "best_val": best,
        "test": test_metrics,
        "ceiling": state_metrics_full(physical, state[test]),
        "decoder": "denso",
        "logp": None,
        "attn_kw": {},
        "history": history,
    }
    return model, checkpoint
