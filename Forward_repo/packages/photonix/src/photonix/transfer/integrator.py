"""Direction-aware thermal transfer, with vertical optical depth at the boundary."""

from __future__ import annotations

import numpy as np

from photonix._validation import finite_vector, positive_scalar, spectral_grid, viewing_cosine
from photonix.constants import DEFAULT_DIFFUSIVITY
from photonix.transfer.planck import planck_radiance_nu


def _prepare_layers(
    wavenumber_cm1, optical_depth_vertical, temperature_k, pressure_hpa, layer_mean
):
    wavenumber = spectral_grid(wavenumber_cm1)
    temperature = finite_vector(temperature_k, "temperature_k", minimum_size=2)
    optical_depth = np.asarray(optical_depth_vertical, dtype=np.float64)
    if optical_depth.shape != (temperature.size - 1, wavenumber.size):
        raise ValueError("optical_depth_vertical must have shape (n_levels - 1, n_wavenumbers)")
    if not np.all(np.isfinite(optical_depth)) or np.any(optical_depth < 0):
        raise ValueError("optical_depth_vertical must be finite and nonnegative")
    if layer_mean not in {"arithmetic", "density"}:
        raise ValueError("layer_mean must be 'arithmetic' or 'density'")
    planck_levels = planck_radiance_nu(wavenumber, temperature)
    planck_mean = 0.5 * (planck_levels[:-1] + planck_levels[1:])
    if pressure_hpa is not None:
        pressure = finite_vector(pressure_hpa, "pressure_hpa")
        if (
            pressure.shape != temperature.shape
            or np.any(pressure <= 0)
            or np.any(np.diff(pressure) <= 0)
        ):
            raise ValueError("pressure_hpa must increase from TOA with one value per temperature")
    elif layer_mean == "density":
        raise ValueError("pressure_hpa is required for layer_mean='density'")
    if layer_mean == "density":
        fraction = (np.arange(10) + 0.5) / 10
        sub_temperature = temperature[:-1, None] + np.diff(temperature)[:, None] * fraction
        sub_pressure = np.exp(
            np.log(pressure[:-1, None]) + np.diff(np.log(pressure))[:, None] * fraction
        )
        density = sub_pressure / sub_temperature
        mean_temperature = np.sum(sub_temperature * density, axis=1) / np.sum(density, axis=1)
        planck_mean = planck_radiance_nu(wavenumber, mean_temperature)
    return optical_depth, planck_levels, planck_mean


def _layer_emission(optical_depth, planck_near, planck_mean):
    """AER EMIN, evaluating thick and thin branches only in their domains."""
    absorbed = -np.expm1(-optical_depth)
    transmittance = np.exp(-optical_depth)
    factor = optical_depth / 6.0
    thick = optical_depth > 1e-3
    depth = optical_depth[thick]
    factor[thick] = 1.0 - 2.0 * (-transmittance[thick] / absorbed[thick] + 1.0 / depth)
    emission = absorbed * (planck_mean + (planck_near - planck_mean) * factor)
    return emission, transmittance


def integrate_upwelling(
    wavenumber_cm1,
    optical_depth_vertical,
    temperature_k,
    *,
    mu_view=1.0,
    pressure_hpa=None,
    layer_mean="arithmetic",
):
    """Return atmospheric upwelling and TOA cumulative transmittance.

    Optical depth is vertical [n_layers,n_nu]; this function applies 1/mu once.
    The cumulative array includes the transparent TOA boundary, shape [n_levels,n_nu].
    """
    mu = viewing_cosine(mu_view)
    optical_depth, planck_levels, planck_mean = _prepare_layers(
        wavenumber_cm1, optical_depth_vertical, temperature_k, pressure_hpa, layer_mean
    )
    slant_depth = optical_depth / mu
    cumulative = np.vstack(
        (np.ones(optical_depth.shape[1]), np.exp(-np.cumsum(slant_depth, axis=0)))
    )
    radiance = np.zeros(optical_depth.shape[1])
    for index in reversed(range(optical_depth.shape[0])):
        emission, transmittance = _layer_emission(
            slant_depth[index], planck_levels[index], planck_mean[index]
        )
        radiance = transmittance * radiance + emission
    return radiance, cumulative


def integrate_downwelling(
    wavenumber_cm1,
    optical_depth_vertical,
    temperature_k,
    *,
    diffusivity=DEFAULT_DIFFUSIVITY,
    pressure_hpa=None,
    layer_mean="arithmetic",
):
    """Return downwelling at the surface; independent of the sensor viewing angle.

    Default diffusivity 1.66 and arithmetic Planck mean match production.
    Density-weighted temperature and 1.67 must be requested explicitly.
    """
    factor = positive_scalar(diffusivity, "diffusivity")
    optical_depth, planck_levels, planck_mean = _prepare_layers(
        wavenumber_cm1, optical_depth_vertical, temperature_k, pressure_hpa, layer_mean
    )
    radiance = np.zeros(optical_depth.shape[1])
    for index in range(optical_depth.shape[0]):
        emission, transmittance = _layer_emission(
            optical_depth[index] * factor, planck_levels[index + 1], planck_mean[index]
        )
        radiance = transmittance * radiance + emission
    return radiance
