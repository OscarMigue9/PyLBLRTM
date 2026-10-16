"""Torch kernels from the forward used to train the final retrieval model.

Inputs use cm^-1, K, atm and molecules/cm^2. Radiances are in
mW m^-2 sr^-1 (cm^-1)^-1. The dense LUT contains log10 cross sections with
dimensions (gas, temperature, log10 pressure, wavenumber). The caller is
responsible for matching the LUT axes, continuum and gas ordering to its data.

The constants and arithmetic below intentionally follow the historical model
kernel. Numerical changes require a separate recipe and checkpoint validation.
"""

from __future__ import annotations

import torch


_C1 = 1.19104297e-16
_C2 = 1.43877688e-2
_MU_DIFF = 1.0 / 1.66


def planck_nu(nu: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """Return Planck radiance for ``nu`` and batchable temperatures."""
    wavenumber_m1 = nu * 100.0
    exponent = _C2 * wavenumber_m1 / temperature.unsqueeze(-1)
    return (_C1 * wavenumber_m1.pow(3) / torch.expm1(exponent)) * 1e5


def bt_nu(nu: torch.Tensor, radiance: torch.Tensor) -> torch.Tensor:
    """Invert :func:`planck_nu` using the historical constants."""
    wavenumber_m1 = nu * 100.0
    return (_C2 * wavenumber_m1) / torch.log1p(_C1 * wavenumber_m1.pow(3) / (radiance * 1e-5))


def interp_sigma(
    logsig: torch.Tensor,
    temperature_axis: torch.Tensor,
    log_pressure_axis: torch.Tensor,
    temperatures: torch.Tensor,
    log_pressures: torch.Tensor,
) -> torch.Tensor:
    """Interpolate log10 cross sections, holding the nearest LUT boundary.

    ``logsig`` has shape (NT, NP, Nnu), and layer states have shape (B, L).
    Clamping both indices and interpolation weights is part of the trained
    model's numerical recipe. The operation remains differentiable in range.
    """
    nt, npres, nnu = logsig.shape
    it = torch.clamp(torch.searchsorted(temperature_axis, temperatures.contiguous()) - 1, 0, nt - 2)
    ip = torch.clamp(
        torch.searchsorted(log_pressure_axis, log_pressures.contiguous()) - 1,
        0,
        npres - 2,
    )
    wt = (
        ((temperatures - temperature_axis[it]) / (temperature_axis[it + 1] - temperature_axis[it]))
        .unsqueeze(-1)
        .clamp(0.0, 1.0)
    )
    wp = (
        (
            (log_pressures - log_pressure_axis[ip])
            / (log_pressure_axis[ip + 1] - log_pressure_axis[ip])
        )
        .unsqueeze(-1)
        .clamp(0.0, 1.0)
    )
    flat = logsig.reshape(nt * npres, nnu)
    c00 = flat[it * npres + ip]
    c01 = flat[it * npres + ip + 1]
    c10 = flat[(it + 1) * npres + ip]
    c11 = flat[(it + 1) * npres + ip + 1]
    interpolated = (
        c00 * (1 - wt) * (1 - wp) + c10 * wt * (1 - wp) + c01 * (1 - wt) * wp + c11 * wt * wp
    )
    return torch.pow(10.0, interpolated)


def _emis(
    optical_depth: torch.Tensor,
    planck_near: torch.Tensor,
    planck_mean: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return EMIN layer emission and transmittance with finite thin-branch gradients."""
    optical_depth = torch.clamp(optical_depth, min=1e-30)
    transmittance = torch.exp(-optical_depth)
    thin = optical_depth <= 1e-3
    safe_transmittance = torch.where(
        torch.abs(transmittance - 1.0) < 1e-15,
        torch.full_like(transmittance, 1.0 - 1e-15),
        transmittance,
    )
    thick_depth = torch.where(thin, torch.ones_like(optical_depth), optical_depth)
    thick_transmittance = torch.where(
        thin, torch.full_like(safe_transmittance, 0.5), safe_transmittance
    )
    factor = torch.where(
        thin,
        optical_depth / 6.0,
        1.0 - 2.0 * (thick_transmittance / (thick_transmittance - 1.0) + 1.0 / thick_depth),
    )
    emission = (1.0 - transmittance) * (planck_mean + (planck_near - planck_mean) * factor)
    return emission, transmittance


def integrate_tud_loop(
    optical_depth: torch.Tensor,
    planck_levels: torch.Tensor,
    mu_up: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate T/U/D with a layer loop for reference and parity checks."""
    planck_top, planck_bottom = planck_levels[:, :-1], planck_levels[:, 1:]
    planck_mean = 0.5 * (planck_top + planck_bottom)
    batch_size, n_layers, nnu = optical_depth.shape
    slant_depth = optical_depth if mu_up is None else optical_depth / mu_up
    upwelling = torch.zeros(batch_size, nnu, dtype=optical_depth.dtype, device=optical_depth.device)
    for layer in range(n_layers - 1, -1, -1):
        emission, transmittance = _emis(
            slant_depth[:, layer], planck_top[:, layer], planck_mean[:, layer]
        )
        upwelling = transmittance * upwelling + emission
    downwelling = torch.zeros(
        batch_size, nnu, dtype=optical_depth.dtype, device=optical_depth.device
    )
    downward_depth = optical_depth / _MU_DIFF
    for layer in range(n_layers):
        emission, transmittance = _emis(
            downward_depth[:, layer], planck_bottom[:, layer], planck_mean[:, layer]
        )
        downwelling = transmittance * downwelling + emission
    total_transmittance = torch.exp(-slant_depth.sum(dim=1))
    return upwelling, downwelling, total_transmittance


def integrate_tud(
    optical_depth: torch.Tensor,
    planck_levels: torch.Tensor,
    mu_up: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Integrate T/U/D over layers using the historical vectorized EMIN formula.

    The viewing cosine affects upwelling and total transmittance. Downwelling
    uses a fixed diffusivity of 1.66, independent of viewing angle.
    """
    planck_top, planck_bottom = planck_levels[:, :-1], planck_levels[:, 1:]
    planck_mean = 0.5 * (planck_top + planck_bottom)
    slant_depth = optical_depth if mu_up is None else optical_depth / mu_up
    up_emission, up_transmittance = _emis(slant_depth, planck_top, planck_mean)
    ones = torch.ones_like(up_transmittance[:, :1])
    preceding = torch.cat([ones, torch.cumprod(up_transmittance, dim=1)[:, :-1]], dim=1)
    upwelling = (up_emission * preceding).sum(dim=1)
    down_emission, down_transmittance = _emis(optical_depth / _MU_DIFF, planck_bottom, planck_mean)
    following_inclusive = torch.flip(torch.cumprod(torch.flip(down_transmittance, [1]), dim=1), [1])
    following = torch.cat([following_inclusive[:, 1:], ones], dim=1)
    downwelling = (down_emission * following).sum(dim=1)
    total_transmittance = torch.exp(-slant_depth.sum(dim=1))
    return upwelling, downwelling, total_transmittance


def forward_lut(
    logsig_gases: torch.Tensor,
    temperature_axis: torch.Tensor,
    log_pressure_axis: torch.Tensor,
    layer_temperatures: torch.Tensor,
    layer_pressures: torch.Tensor,
    gas_columns: torch.Tensor,
    continuum_depth: torch.Tensor | float,
    level_temperatures: torch.Tensor,
    skin_temperature: torch.Tensor,
    emissivity: torch.Tensor | float,
    wavenumber: torch.Tensor,
    checkpoint: bool = False,
    mu: torch.Tensor | float | None = None,
) -> dict[str, torch.Tensor]:
    """Return T/U/D and sensor radiance from a dense multi-gas LUT.

    The caller supplies continuum depth explicitly. Surface composition and
    radiance units follow the historical training kernel. A non-tensor continuum
    input selects zero continuum, as in that kernel; pass a tensor for nonzero
    optical depth.
    """
    log_pressures = torch.log10(layer_pressures)
    batch_size, n_layers = layer_temperatures.shape
    optical_depth = (
        continuum_depth
        if torch.is_tensor(continuum_depth)
        else torch.zeros(
            batch_size,
            n_layers,
            wavenumber.shape[0],
            dtype=layer_temperatures.dtype,
            device=layer_temperatures.device,
        )
    )
    for gas in range(logsig_gases.shape[0]):
        sigma = interp_sigma(
            logsig_gases[gas],
            temperature_axis,
            log_pressure_axis,
            layer_temperatures,
            log_pressures,
        )
        optical_depth = optical_depth + sigma * gas_columns[:, gas].unsqueeze(-1)
    planck_levels = planck_nu(wavenumber, level_temperatures)
    mu_up = mu.view(-1, 1, 1) if torch.is_tensor(mu) else mu
    if checkpoint:
        from torch.utils.checkpoint import checkpoint as torch_checkpoint

        upwelling, downwelling, transmittance = torch_checkpoint(
            integrate_tud, optical_depth, planck_levels, mu_up, use_reentrant=False
        )
    else:
        upwelling, downwelling, transmittance = integrate_tud(optical_depth, planck_levels, mu_up)
    skin_planck = planck_nu(wavenumber, skin_temperature)
    if torch.is_tensor(emissivity) and emissivity.dim() == 1:
        emissivity = emissivity.unsqueeze(-1)
    observed = (
        emissivity * skin_planck * transmittance
        + upwelling
        + (1.0 - emissivity) * transmittance * downwelling
    )
    return {"T": transmittance, "U": upwelling, "D": downwelling, "L_obs": observed}


def vmr_to_ncol(vmr: torch.Tensor, air_column: torch.Tensor) -> torch.Tensor:
    """Convert gas VMR and dry-air column to molecules/cm^2 per layer."""
    return vmr * air_column.unsqueeze(-2)
