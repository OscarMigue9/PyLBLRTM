"""Final-state adapter to the versioned historical Torch forward.

Source: ``model`` commit 9f3e0a4, ``_common.py::PhysForward`` in full/Ncol
mode with the configuration in ``cadena_sew714.sh``. The numerical kernels are
provided by :mod:`photonix.torch_legacy`. The LUT, continuum and pressure
correction assets must be resolved and verified by the artifact manifest.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from photonix.torch_legacy import (
    continuum_od_torch,
    integrate_tud,
    interp_sigma,
    planck_nu,
    prep_continuum_tables,
)

from .state import GASES, N_LAYERS, N_LEVELS, SKIN_INDEX, STATE_DIM, gas_slice

NU0 = 714.0
NU1 = 1250.0
DNU = 0.001
DNU_CHANNEL = 0.25
FWHM_CHANNEL = 0.5
WINDOW_WIDTH = 25.0
WINDOW_MARGIN = 3.0
LUT_NU0 = 635.0
LUT_NU1 = 1265.0
LUT_P_LO = 1.4e-5
LUT_P_HI = 1.189
LUT_NP = 145
T_AXIS = np.linspace(170, 330, 20)
NU_FINE = np.arange(NU0, NU1 + DNU / 2, DNU)
NU_CHANNEL = np.arange(NU0, NU1 + DNU_CHANNEL / 2, DNU_CHANNEL)
LOGP_AXIS = np.log10(np.logspace(np.log10(LUT_P_LO), np.log10(LUT_P_HI), LUT_NP))
SIGMA_CHANNEL = FWHM_CHANNEL / (2 * np.sqrt(2 * np.log(2)))
VMR_REF = {"H2O": 1.0e-2, "CO2": 4.0e-4, "O3": 2.66e-8, "N2O": 3.2e-7, "CH4": 1.7e-6}


class PhysicalForward:
    """Differentiable radiance from the 818 component Ncol state.

    This class reads five dense LUT files and therefore belongs on the Modal GPU
    path. It deliberately requires the pressure correction file used to build
    the final training residual; an absent asset must fail visibly.
    """

    MAX_WINDOW_CACHE = 64

    def __init__(
        self,
        pressure_layers: np.ndarray,
        emissivity: np.ndarray,
        vmr_ratio: np.ndarray,
        lut_dir: str | Path,
        r_self_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.pressure_layers = np.asarray(pressure_layers, dtype=np.float32)
        self.emissivity = np.asarray(emissivity, dtype=np.float64)
        if self.pressure_layers.ndim != 2 or self.pressure_layers.shape[1] != N_LAYERS:
            raise ValueError("Pressure layers must have shape (N, 136)")
        if self.emissivity.shape != (len(self.pressure_layers), len(NU_CHANNEL)):
            raise ValueError("Emissivity must have shape (N, 2145)")
        ratio = np.asarray(vmr_ratio, dtype=np.float32)
        if ratio.shape != (N_LAYERS,):
            raise ValueError("Historical water VMR/Ncol ratio must have 136 layers")
        self.vmr_ratio = torch.as_tensor(ratio, device=self.device)[None, :]
        self.temperature_axis = torch.as_tensor(T_AXIS, dtype=torch.float32, device=self.device)
        self.log_pressure_axis = torch.as_tensor(LOGP_AXIS, dtype=torch.float32, device=self.device)
        self.continuum = prep_continuum_tables(NU_FINE)
        self.window_cache: dict[float, tuple] = {}

        lut_points = int(round((LUT_NU1 - LUT_NU0) / DNU)) + 1
        first = int(round((NU0 - LUT_NU0) / DNU))
        last = first + len(NU_FINE)
        cubes = []
        for gas in GASES:
            path = Path(lut_dir) / f"lut_band645_{gas}.npy"
            cube = np.load(path, mmap_mode="r")
            if cube.size != len(T_AXIS) * LUT_NP * lut_points:
                raise ValueError(f"{path} does not match the 20x145x630001 LUT recipe")
            selected = np.asarray(cube.reshape(len(T_AXIS), LUT_NP, lut_points)[:, :, first:last])
            cubes.append(torch.from_numpy(selected.copy()).float())
        self.lut = torch.stack(cubes).to(self.device)

        with np.load(r_self_path) as correction:
            self.r_self = {
                gas: torch.as_tensor(correction[gas], dtype=torch.float32, device=self.device)
                for gas in GASES
            }
            self.r_self_edges = np.asarray(correction["bordes"], dtype=np.float64)

    @classmethod
    def from_dataset(
        cls,
        dataset: Mapping[str, object],
        lut_dir: str | Path,
        r_self_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "PhysicalForward":
        """Fit the historical climatological water ratio on training rows."""
        data = dataset["d"]
        train = np.asarray(dataset["idx"]["train"])
        water_vmr = np.asarray(data["VMR_H2O"][train], dtype=np.float64).mean(axis=0)
        water_column = np.asarray(data["Ncol"][train, 0], dtype=np.float64).mean(axis=0)
        ratio = water_vmr / np.maximum(water_column, 1.0e5)
        return cls(dataset["P_lay"], dataset["EPS"], ratio, lut_dir, r_self_path, device=device)

    def with_scene_tables(
        self, pressure_layers: np.ndarray, emissivity: np.ndarray
    ) -> "PhysicalForward":
        """Bind new row tables while sharing the resident, read-only LUT tensors."""
        pressure = np.asarray(pressure_layers, dtype=np.float32)
        surface = np.asarray(emissivity, dtype=np.float64)
        if pressure.ndim != 2 or pressure.shape[1] != N_LAYERS:
            raise ValueError("Scene pressure layers must have shape (N, 136)")
        if surface.shape != (len(pressure), len(NU_CHANNEL)):
            raise ValueError("Scene emissivity must have shape (N, 2145)")
        if not np.isfinite(pressure).all() or not np.isfinite(surface).all():
            raise ValueError("Scene forward tables must be finite")
        bound = copy.copy(self)
        bound.pressure_layers = pressure.copy()
        bound.emissivity = surface.copy()
        bound.window_cache = {}
        return bound

    def _pressure_effective(
        self, gas: str, pressure: torch.Tensor, vmr: torch.Tensor, nu_lo: float
    ) -> torch.Tensor:
        block = int(
            np.clip(
                np.searchsorted(self.r_self_edges, nu_lo - 1.0e-6) - 1, 0, len(self.r_self[gas]) - 1
            )
        )
        ratio = self.r_self[gas][block]
        reference = VMR_REF[gas]
        return pressure * (1.0 + (ratio - 1.0) * vmr) / (1.0 + (ratio - 1.0) * reference)

    def _win(self, lo: float) -> tuple:
        """Return cached window grids, ISRF matrix and continuum tables."""
        key = round(lo, 6)
        if key in self.window_cache:
            return self.window_cache[key]
        if len(self.window_cache) >= self.MAX_WINDOW_CACHE:
            self.window_cache.clear()
        hi = lo + WINDOW_WIDTH
        first = int(round((lo - NU0) / DNU))
        last = int(round((hi - NU0) / DNU)) + 1
        mask = (NU_CHANNEL >= lo + WINDOW_MARGIN) & (NU_CHANNEL <= hi - WINDOW_MARGIN)
        channel_nu = NU_CHANNEL[mask]
        weights = np.exp(
            -0.5 * ((channel_nu[:, None] - NU_FINE[None, first:last]) / SIGMA_CHANNEL) ** 2
        )
        weights = weights / weights.sum(axis=1, keepdims=True)
        cs, cf, exponent = self.continuum
        result = (
            first,
            last,
            mask,
            torch.as_tensor(weights, dtype=torch.float32, device=self.device),
            torch.as_tensor(NU_FINE[first:last], dtype=torch.float32, device=self.device),
            torch.as_tensor(channel_nu, dtype=torch.float32, device=self.device),
            torch.as_tensor(cs[first:last], dtype=torch.float32, device=self.device),
            torch.as_tensor(cf[first:last], dtype=torch.float32, device=self.device),
            torch.as_tensor(exponent[first:last], dtype=torch.float32, device=self.device),
        )
        self.window_cache[key] = result
        return result

    def window(
        self,
        state: torch.Tensor,
        rows: np.ndarray,
        lo: float,
        mu: torch.Tensor | None = None,
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, np.ndarray]:
        """Render radiance for one 25 cm⁻¹ window using the historical formula."""
        first, last, mask, weights, fine_nu, channel_nu, cs, cf, exponent = self._win(lo)
        if mask.sum() < 4:
            return None, mask
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError("Forward state must have shape (N, 818)")
        temperature = state[:, :N_LEVELS]
        skin = state[:, SKIN_INDEX]
        columns = torch.stack([torch.pow(10.0, state[:, gas_slice(gas)]) for gas in GASES], dim=1)
        layer_temperature = 0.5 * (temperature[:, :-1] + temperature[:, 1:])
        pressure = torch.as_tensor(self.pressure_layers[rows], device=self.device)
        water_column = columns[:, 0]
        air_column = torch.clamp(
            water_column / torch.clamp(self.vmr_ratio * water_column, min=1.0e-30), min=1.0e-30
        )
        water_vmr = water_column / air_column
        optical_depth = continuum_od_torch(
            fine_nu, cs, cf, exponent, layer_temperature, pressure, water_vmr, water_column
        )
        for index, gas in enumerate(GASES):
            gas_pressure = self._pressure_effective(
                gas, pressure, columns[:, index] / torch.clamp(air_column, min=1.0e-30), lo
            )
            section = interp_sigma(
                self.lut[index, :, :, first:last],
                self.temperature_axis,
                self.log_pressure_axis,
                layer_temperature,
                torch.log10(torch.clamp(gas_pressure, min=1.0e-30)),
            )
            optical_depth = optical_depth + section * columns[:, index].unsqueeze(-1)
        mu_up = None
        if mu is not None:
            mu_tensor = (
                mu
                if torch.is_tensor(mu)
                else torch.as_tensor(mu, dtype=torch.float32, device=self.device)
            )
            mu_up = mu_tensor.reshape(-1, 1, 1) if mu_tensor.ndim > 0 else mu_tensor
        up, down, transmission = integrate_tud(
            optical_depth, planck_nu(fine_nu, temperature), mu_up
        )
        transmission_channel = transmission @ weights.T
        up_channel = up @ weights.T
        down_channel = down @ weights.T
        if eps is None:
            eps = torch.as_tensor(
                self.emissivity[rows][:, mask], dtype=torch.float32, device=self.device
            )
        surface = planck_nu(channel_nu, skin)
        radiance = (
            eps * surface * transmission_channel
            + up_channel
            + (1 - eps) * transmission_channel * down_channel
        )
        return radiance, mask
