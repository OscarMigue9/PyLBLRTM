"""Layer-resolved AER line optical depth, with a single owner for every sample."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np

from photonix.constants import (
    DOPPLER_COEFFICIENT,
    REFERENCE_TEMPERATURE_K,
    SECOND_RADIATION_CONSTANT_CM_K,
)
from .aer import AerLineMixing, aer_line_profile, coupling_at_temperature
from .continuum import _radiation_term
from .hitran import load_partition_function


def _spectral_grid(wavenumber_cm1, tile_width_cm1, line_cutoff_cm1):
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    if nu.ndim != 1 or nu.size < 2 or not np.isfinite(nu).all() or np.any(nu <= 0):
        raise ValueError("Line optical depth requires at least two finite positive frequencies")
    differences = np.diff(nu)
    step = float(differences[0])
    if step <= 0 or not np.allclose(differences, step, rtol=1e-7, atol=1e-10):
        raise ValueError("Line optical-depth grid must be increasing and uniformly spaced")
    if (
        not np.isfinite(tile_width_cm1)
        or tile_width_cm1 <= step
        or np.isclose(tile_width_cm1, step, rtol=1e-8, atol=1e-12)
    ):
        raise ValueError("tile_width_cm1 must be finite and greater than the spectral step")
    if not np.isfinite(line_cutoff_cm1) or line_cutoff_cm1 <= 0:
        raise ValueError("line_cutoff_cm1 must be finite and positive")
    # Integer index partitions, independent of floating-point equality at boundaries.
    samples_per_tile = max(1, int(np.floor(tile_width_cm1 / step + 1e-8)))
    return nu, samples_per_tile


def _layer_inputs(layer, species):
    temperature = float(layer.temperature_k)
    pressure = float(layer.pressure_atm)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"Layer {layer.index}: temperature must be finite and positive")
    if not np.isfinite(pressure) or pressure <= 0:
        raise ValueError(f"Layer {layer.index}: pressure must be finite and positive")
    for gas in species:
        column = float(layer.column_density_cm2.get(gas.name, 0.0))
        vmr = float(layer.vmr.get(gas.name, 0.0))
        if not np.isfinite(column) or column < 0 or not np.isfinite(vmr) or not 0 <= vmr <= 1:
            raise ValueError(f"Layer {layer.index}, {gas.name}: invalid column density or VMR")
        if column > 0 and gas.name not in layer.vmr:
            raise ValueError(f"Layer {layer.index}, {gas.name}: nonzero column requires VMR")
    return temperature, pressure


def build_optical_depth_stack(
    wavenumber_cm1,
    lines,
    layers,
    species,
    *,
    line_mixing: AerLineMixing | None = None,
    voigt_method: str = "humlicek",
    tile_width_cm1: float = 2.0,
    line_cutoff_cm1: float = 25.0,
) -> np.ndarray:
    """Compute float64 vertical line OD [layer, frequency], excluding all continua.

    HITRAN strengths already encode isotopic abundance; every selected isotope uses
    its gas's full column. Species/line masks, AER associations and partition ratios
    are prepared once before the layer/tile loops. Per-layer pressure shifts are
    used, correcting the frozen GPU forward's reuse of the first layer's centers.
    The caller owns the physical output-band restriction; this backend accepts the
    padded ILS domain. Signed AER mixing terms are not clipped before continuum sum.
    The radiation term nu*tanh(c2*nu/2T) is divided out at each shifted line centre
    and applied at every frequency, as LBLRTM does with JRAD=0 (SpectralCalc Eq. 11),
    so line wings scale as R(nu, T)/R(nu_c, T) instead of 1.
    """
    nu, tile_samples = _spectral_grid(wavenumber_cm1, tile_width_cm1, line_cutoff_cm1)
    if voigt_method not in {"humlicek", "scipy"}:
        raise ValueError("voigt_method must be 'humlicek' or 'scipy'")
    layers, species = tuple(layers), tuple(species)
    conditions = [_layer_inputs(layer, species) for layer in layers]
    result = np.zeros((len(layers), nu.size), dtype=np.float64)
    if not layers:
        return result
    if len({gas.name for gas in species}) != len(species):
        raise ValueError("Duplicate gas names in species selection")
    maximum_pressure = max(pressure for _, pressure in conditions)
    available_pairs = lines.isotope_pairs
    shift_bound = np.abs(lines.pressure_shift_cm1_atm) * maximum_pressure
    prepared = []
    selected_pairs = set()
    for gas in species:
        active_layers = [float(layer.column_density_cm2.get(gas.name, 0.0)) > 0 for layer in layers]
        if not any(active_layers):
            continue
        absent = [
            isotope.isotope_id
            for isotope in gas.isotopologues
            if (gas.molecule_id, isotope.isotope_id) not in available_pairs
        ]
        if len(absent) == len(gas.isotopologues):
            raise ValueError(
                f"{gas.name}: no selected isotopologue is present in the input line list"
            )
        if absent:
            warnings.warn(
                f"{gas.name}: selected isotopologues absent from the input line list: {absent}; "
                "their line absorption is omitted",
                RuntimeWarning,
                stacklevel=2,
            )
        for isotope in gas.isotopologues:
            pair = (gas.molecule_id, isotope.isotope_id)
            if pair in selected_pairs:
                raise ValueError(f"Duplicate isotope selection {pair}")
            selected_pairs.add(pair)
            # Bound the pressure shift for every layer before discarding remote lines.
            mask = (
                (lines.molecule_id == gas.molecule_id)
                & (lines.isotope_id == isotope.isotope_id)
                & (lines.wavenumber_cm1 >= nu[0] - line_cutoff_cm1 - shift_bound)
                & (lines.wavenumber_cm1 <= nu[-1] + line_cutoff_cm1 + shift_bound)
            )
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            coefficients = None
            if gas.molecule_id == 2:
                if not isinstance(line_mixing, AerLineMixing):
                    raise ValueError("AER CO2 absorption requires explicit AerLineMixing input")
                subset = SimpleNamespace(
                    molecule_id=lines.molecule_id[indices],
                    isotope_id=lines.isotope_id[indices],
                    wavenumber_cm1=lines.wavenumber_cm1[indices],
                )
                coefficients = line_mixing.coefficients_for(subset)
            q_ratios = np.ones(len(layers), dtype=np.float64)
            for k, ((temperature, _), active) in enumerate(zip(conditions, active_layers)):
                if active:
                    q_reference, q_temperature = load_partition_function(
                        gas.name, isotope, temperature
                    )
                    if (
                        not np.isfinite(q_reference)
                        or not np.isfinite(q_temperature)
                        or q_reference <= 0
                        or q_temperature <= 0
                    ):
                        raise ValueError(
                            f"Invalid partition function for {gas.name}/{isotope.isotope_id}"
                        )
                    q_ratios[k] = q_reference / q_temperature
            prepared.append((gas, isotope, indices, coefficients, q_ratios))

    c2 = SECOND_RADIATION_CONSTANT_CM_K
    reference_temperature = REFERENCE_TEMPERATURE_K
    for k, (layer, (temperature, pressure)) in enumerate(zip(layers, conditions)):
        for gas, isotope, indices, coefficients, q_ratios in prepared:
            column = float(layer.column_density_cm2.get(gas.name, 0.0))
            if column == 0:
                continue
            self_pressure = float(layer.vmr[gas.name]) * pressure
            air_pressure = pressure - self_pressure
            centers = lines.wavenumber_cm1[indices]
            shifted_centers = centers + lines.pressure_shift_cm1_atm[indices] * air_pressure
            lower_energy = lines.lower_energy_cm1[indices]
            # A difference of inverse temperatures avoids the original exp/exp underflow.
            # The line-centre radiation term is removed here and restored per frequency.
            strength = (
                lines.line_strength_ref[indices]
                * q_ratios[k]
                * np.exp(-c2 * lower_energy * (1.0 / temperature - 1.0 / reference_temperature))
                * (-np.expm1(-c2 * centers / temperature))
                / (-np.expm1(-c2 * centers / reference_temperature))
                * column
                / _radiation_term(shifted_centers, temperature)
            )
            lorentz = (
                (reference_temperature / temperature) ** lines.temperature_exponent[indices]
            ) * (
                lines.air_width_cm1_atm[indices] * air_pressure
                + lines.self_width_cm1_atm[indices] * self_pressure
            )
            doppler = (
                centers * DOPPLER_COEFFICIENT * np.sqrt(temperature / isotope.molar_mass_g_mol)
            )
            coupling = (
                coupling_at_temperature(coefficients, temperature)
                if coefficients is not None
                else None
            )
            if not np.isfinite(strength).all() or np.any(strength < 0):
                raise FloatingPointError(f"Layer {layer.index}, {gas.name}: invalid line strength")
            for start in range(0, nu.size, tile_samples):
                stop = min(start + tile_samples, nu.size)
                tile = nu[start:stop]
                nearby = (shifted_centers >= tile[0] - line_cutoff_cm1) & (
                    shifted_centers <= tile[-1] + line_cutoff_cm1
                )
                if not nearby.any():
                    continue
                # Bound temporary profile matrices for dense line lists.
                nearby_indices = np.flatnonzero(nearby)
                for block_start in range(0, nearby_indices.size, 256):
                    block = nearby_indices[block_start : block_start + 256]
                    profile = aer_line_profile(
                        tile,
                        shifted_centers[block],
                        doppler[block],
                        lorentz[block],
                        is_co2=gas.molecule_id == 2,
                        coupling=coupling[block] if coupling is not None else None,
                        pressure_atm=pressure,
                        cutoff_cm1=line_cutoff_cm1,
                        voigt_method=voigt_method,
                    )
                    result[k, start:stop] += profile @ strength[block]
        result[k] *= _radiation_term(nu, temperature)
        if not np.isfinite(result[k]).all():
            raise FloatingPointError(f"Layer {layer.index}: non-finite AER optical depth")
    return result
