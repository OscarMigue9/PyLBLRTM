"""Historical differentiable H2O continuum used by the final retrieval model.

The small frozen coefficient table is loaded from this installed package. This
keeps the numerical source used during training without the old local ``.pyc``
fallback or absolute Windows path.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch


@lru_cache(maxsize=1)
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    resource = files("photonix.torch_legacy").joinpath("continuum_tables.npz")
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as archive:
        return (
            archive["NU_TABLE"].astype(np.float64),
            archive["CS_REF"].astype(np.float64),
            archive["CF_REF"].astype(np.float64),
            archive["TEXP_REF"].astype(np.float64),
            float(archive["T_REF"]),
            float(archive["P_REF_HP"]),
            float(archive["C2"]),
        )


def prep_continuum_tables(
    wavenumber_cm1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate H2O self/foreign coefficients and exponent onto a fine grid.

    Log interpolation and held endpoint values reproduce the trained model.
    Callers should still use the validated spectral coverage of their dataset.
    """
    nu_axis, self_ref, foreign_ref, exponent_ref, _, _, _ = _tables()
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    log_self = np.interp(
        nu,
        nu_axis,
        np.log(self_ref),
        left=np.log(self_ref[0]),
        right=np.log(self_ref[-1]),
    )
    log_foreign = np.interp(
        nu,
        nu_axis,
        np.log(foreign_ref),
        left=np.log(foreign_ref[0]),
        right=np.log(foreign_ref[-1]),
    )
    exponent = np.interp(nu, nu_axis, exponent_ref, left=exponent_ref[0], right=exponent_ref[-1])
    return np.exp(log_self), np.exp(log_foreign), exponent


def continuum_od_torch(
    wavenumber_cm1: torch.Tensor,
    self_coefficient: torch.Tensor,
    foreign_coefficient: torch.Tensor,
    temperature_exponent: torch.Tensor,
    layer_temperatures: torch.Tensor,
    layer_pressures_atm: torch.Tensor,
    water_vmr: torch.Tensor,
    water_columns: torch.Tensor,
) -> torch.Tensor:
    """Return differentiable H2O continuum optical depth per layer."""
    _, _, _, _, reference_temperature, reference_pressure, radiation_constant = _tables()
    nu = wavenumber_cm1.view(1, 1, -1)
    pressure_hpa = layer_pressures_atm * 1013.25
    density_ratio = (pressure_hpa / reference_pressure) * (
        reference_temperature / layer_temperatures
    )
    argument = nu * (radiation_constant / layer_temperatures.unsqueeze(-1))
    radiation = torch.where(
        argument <= 0.01,
        0.5 * argument * nu,
        torch.where(
            argument <= 10.0,
            nu * (1.0 - torch.exp(-argument)) / (1.0 + torch.exp(-argument)),
            nu,
        ),
    )
    ratio = density_ratio.unsqueeze(-1)
    self_term = (
        self_coefficient.view(1, 1, -1)
        * (reference_temperature / layer_temperatures.unsqueeze(-1)).pow(
            temperature_exponent.view(1, 1, -1)
        )
        * water_vmr.unsqueeze(-1)
        * ratio
        * radiation
    )
    foreign_term = (
        foreign_coefficient.view(1, 1, -1) * (1.0 - water_vmr.unsqueeze(-1)) * ratio * radiation
    )
    return torch.clamp((self_term + foreign_term) * water_columns.unsqueeze(-1), min=0.0)
