"""The published full state layout and metrics.

Source: ``model`` commit 9f3e0a4, ``_common.py``. Gas values are log10 layer
columns (molecules/cm²); temperature values are kelvin. The exact ordering,
column floor, standardization and block loss are checkpoint contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
N_LEVELS = 137
N_LAYERS = 136
STATE_DIM = 818
SKIN_INDEX = 817
NCOL_FLOOR = 1.0e6
SEED_SPLIT = 42
SIGMA_VMR_DEX = {"H2O": 0.3777, "CO2": 0.0045, "O3": 0.2620, "N2O": 0.1855, "CH4": 0.0786}


def gas_slice(gas: str) -> slice:
    """Return the slice occupied by one gas in the 818 component state."""
    index = GASES.index(gas)
    start = N_LEVELS + index * N_LAYERS
    return slice(start, start + N_LAYERS)


BLOCKS = (
    ("T", slice(0, N_LEVELS)),
    *((gas, gas_slice(gas)) for gas in GASES),
    ("T_skin", slice(SKIN_INDEX, STATE_DIM)),
)


def build_state_matrix_full(data: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build the exact physical Ncol state expected by the final decoder."""
    columns = np.asarray(data["Ncol"], dtype=np.float64)
    temperatures = np.asarray(data["T_levels"], dtype=np.float64)
    skin = np.asarray(data["T_skin"], dtype=np.float64)
    if columns.ndim != 3 or columns.shape[1:] != (len(GASES), N_LAYERS):
        raise ValueError(f"Ncol must have shape (N, 5, 136), got {columns.shape}")
    if temperatures.shape != (len(columns), N_LEVELS):
        raise ValueError(f"T_levels must have shape (N, 137), got {temperatures.shape}")
    if skin.shape != (len(columns),):
        raise ValueError(f"T_skin must have shape (N,), got {skin.shape}")
    parts = [temperatures]
    parts.extend(
        np.log10(np.clip(columns[:, index], NCOL_FLOOR, None)) for index in range(len(GASES))
    )
    parts.append(skin[:, None])
    return np.concatenate(parts, axis=1)


def split_indices(n_profiles: int = 30_000) -> dict[str, np.ndarray]:
    """Materialize the historical 24k/3k/3k split from NumPy seed 42."""
    if n_profiles != 30_000:
        raise ValueError("The published split requires exactly 30,000 profiles")
    permutation = np.random.default_rng(SEED_SPLIT).permutation(n_profiles)
    return {
        "train": permutation[:24_000],
        "val": permutation[24_000:27_000],
        "test": permutation[27_000:],
    }


@dataclass
class Standardizer:
    """Historical mean and population standard deviation with a 1e-9 floor."""

    mu: np.ndarray | None = None
    sd: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        self.mu = values.mean(axis=0)
        self.sd = values.std(axis=0) + 1.0e-9
        return self

    def to_z(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mu) / self.sd

    def to_phys(self, values: np.ndarray) -> np.ndarray:
        return values * self.sd + self.mu

    def state_dict(self) -> dict[str, np.ndarray]:
        return {"mu": np.asarray(self.mu), "sd": np.asarray(self.sd)}

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> "Standardizer":
        return cls(np.asarray(state["mu"]), np.asarray(state["sd"]))


def block_mse(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Average the MSE of each of the seven physical state blocks."""
    if pred.shape != true.shape or pred.shape[-1] != STATE_DIM:
        raise ValueError("Both state tensors must have matching shape (N, 818)")
    return torch.stack(
        [torch.mean((pred[:, span] - true[:, span]) ** 2) for _, span in BLOCKS]
    ).mean()


def state_metrics_full(
    predicted: np.ndarray,
    true: np.ndarray,
    n_air: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the metrics reported by the historical full-state evaluation."""
    predicted = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if predicted.shape != true.shape or predicted.ndim != 2 or predicted.shape[1] != STATE_DIM:
        raise ValueError("States must have matching shape (N, 818)")
    result = {
        "rmse_T_K": float(np.sqrt(np.mean((predicted[:, :N_LEVELS] - true[:, :N_LEVELS]) ** 2))),
        "rmse_Tskin_K": float(
            np.sqrt(np.mean((predicted[:, SKIN_INDEX] - true[:, SKIN_INDEX]) ** 2))
        ),
    }
    for gas in GASES:
        span = gas_slice(gas)
        delta = predicted[:, span] - true[:, span]
        result[f"rmse_{gas}_dex"] = float(np.sqrt(np.mean(delta**2)))
        result[f"bias_{gas}_dex"] = float(np.mean(delta))
        predicted_column = np.log10(np.power(10.0, predicted[:, span]).sum(axis=1))
        true_column = np.log10(np.power(10.0, true[:, span]).sum(axis=1))
        result[f"col{gas}_dex"] = float(np.sqrt(np.mean((predicted_column - true_column) ** 2)))
        predicted_shape = predicted[:, span] - predicted_column[:, None]
        true_shape = true[:, span] - true_column[:, None]
        result[f"forma_{gas}_dex"] = float(np.sqrt(np.mean((predicted_shape - true_shape) ** 2)))
        result[f"skill_{gas}"] = float(1.0 - (result[f"rmse_{gas}_dex"] / SIGMA_VMR_DEX[gas]) ** 2)
        if n_air is not None:
            air_column = np.asarray(n_air).sum(axis=1)
            predicted_mixing = np.power(10.0, predicted[:, span]).sum(axis=1) / air_column
            true_mixing = np.power(10.0, true[:, span]).sum(axis=1) / air_column
            result[f"X{gas}_rel"] = float(
                np.sqrt(np.mean(((predicted_mixing - true_mixing) / true_mixing) ** 2))
            )
            if gas == "CO2":
                result["XCO2_ppm"] = float(
                    np.sqrt(np.mean(((predicted_mixing - true_mixing) * 1e6) ** 2))
                )
    return result
