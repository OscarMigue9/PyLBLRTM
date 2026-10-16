"""Public AER forward: atmosphere-only T/U/D followed by surface radiance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from photonix._validation import positive_scalar, viewing_cosine
from photonix.atmosphere import AtmosphericProfile
from photonix.constants import BAND_MAX_CM1, BAND_MIN_CM1, DEFAULT_DIFFUSIVITY, RECIPE_ID
from photonix.lut import CrossSectionLUT
from photonix.spectroscopy.absorption import build_optical_depth_stack
from photonix.spectroscopy.aer import AerLineMixing
from photonix.spectroscopy.continuum import build_continuum_optical_depth
from photonix.spectroscopy.hitran import LineList, read_hitran_par
from photonix.spectroscopy.species import Species, default_species
from photonix.transfer.ils import InstrumentLineShape, apply_ils_radiance
from photonix.transfer.integrator import integrate_downwelling, integrate_upwelling
from photonix.transfer.planck import planck_radiance_nu


@dataclass(frozen=True)
class NadirForwardOutput:
    """Atmosphere-only result on the internal (possibly padded) fine grid.

    Optical depth is vertical, float64 and pre-ILS. Radiances are mW m^-2 sr^-1
    (cm^-1)^-1. The requested output grid is distinct from the internal grid.
    """

    wavenumber_cm1: np.ndarray
    output_wavenumber_cm1: np.ndarray
    upwelling_radiance: np.ndarray
    downwelling_radiance: np.ndarray
    cumulative_transmittance: np.ndarray
    optical_depth_vertical: np.ndarray
    mu_view: float
    ils: InstrumentLineShape | None
    backend: str
    metadata: dict

    @property
    def total_transmittance(self) -> np.ndarray:
        """Atmospheric column transmittance along the viewing path."""
        return self.cumulative_transmittance[-1]


def run_nadir_simulation(
    profile: AtmosphericProfile,
    species: Sequence[Species] | None = None,
    *,
    parfile: str | Path | None = None,
    record_format: str = "hitran160",
    lines: LineList | None = None,
    lut: CrossSectionLUT | None = None,
    line_mixing: AerLineMixing | None = None,
    nu_min_cm1: float = BAND_MIN_CM1,
    nu_max_cm1: float = BAND_MAX_CM1,
    step_cm1: float = 0.001,
    mu_view: float = 1.0,
    ils: InstrumentLineShape | None = None,
    include_continuum: bool = True,
    diffusivity: float = DEFAULT_DIFFUSIVITY,
    layer_mean: str = "arithmetic",
    voigt_method: str = "humlicek",
    tile_width_cm1: float = 2.0,
    line_cutoff_cm1: float = 25.0,
) -> NadirForwardOutput:
    """Calculate vertical optical depth and integrate T/U/D for one atmosphere.

    Supply exactly one of parfile, lines, or lut. The public band is 714–1250
    cm^-1; ILS margin is calculated internally. ``species=None`` uses all
    supported isotopologues of the gases present in the profile. Use
    ``default_species(..., isotopologues=[(1, '1'), ...])`` for an explicit subset.

    For a local AER 100-column parfile select record_format='aer100'; the default
    requires HITRAN's 160-column records. AER CO2 requires explicit coupling data
    for LBL. A LUT carries its own
    isotopologue and continuum contract; incompatible requests are errors.
    """
    if not isinstance(profile, AtmosphericProfile):
        raise TypeError("profile must be an AtmosphericProfile")
    if sum(value is not None for value in (parfile, lines, lut)) != 1:
        raise ValueError("supply exactly one of parfile, lines, or lut")
    if record_format not in {"hitran160", "aer100"}:
        raise ValueError("record_format must be 'hitran160' or 'aer100'")
    if parfile is None and record_format != "hitran160":
        raise ValueError("record_format is only used when parfile is supplied")
    low = positive_scalar(nu_min_cm1, "nu_min_cm1")
    high = positive_scalar(nu_max_cm1, "nu_max_cm1")
    step = positive_scalar(step_cm1, "step_cm1")
    mu = viewing_cosine(mu_view)
    if not BAND_MIN_CM1 <= low < high <= BAND_MAX_CM1:
        raise ValueError("the thermal AER recipe supports 714 <= nu_min_cm1 < nu_max_cm1 <= 1250")
    if ils is not None and not isinstance(ils, InstrumentLineShape):
        raise TypeError("ils must be an InstrumentLineShape or None")
    selected = tuple(default_species(tuple(profile.vmr)) if species is None else species)
    layers = profile.build_layers(selected)
    padding_samples = 0 if ils is None else ils.half_width_samples(step)
    n_intervals = int(np.ceil((high - low) / step - 1e-9))
    if n_intervals < 1:
        raise ValueError("step_cm1 must resolve the requested interval")
    wavenumber = low + np.arange(-padding_samples, n_intervals + padding_samples + 1) * step
    output_step = step if ils is None else ils.output_step_cm1
    n_output = int(np.floor((high - low) / output_step + 1e-9)) + 1
    output_grid = low + np.arange(n_output) * output_step
    if lut is not None:
        if line_mixing is not None:
            raise ValueError("line_mixing is already encoded in the LUT; supply it only to LBL")
        if voigt_method != "humlicek" or line_cutoff_cm1 != 25.0 or tile_width_cm1 != 2.0:
            raise ValueError(
                "voigt_method, line_cutoff_cm1 and tile_width_cm1 are LBL-only options; "
                "the LUT uses its stored spectroscopy"
            )
        if lut.recipe != RECIPE_ID:
            raise ValueError(
                f"LUT recipe {lut.recipe!r} is incompatible with runtime recipe {RECIPE_ID!r}"
            )
        if lut.includes_h2o_continuum:
            raise ValueError("this recipe requires water continuum outside the aggregated LUT")
        if (
            not include_continuum
            and lut.includes_co2_continuum
            and any(gas.name == "CO2" for gas in selected)
        ):
            raise ValueError("cannot disable continuum already embedded in the CO2 LUT")
        optical_depth = lut.optical_depth(wavenumber, layers, selected)
        add_co2_continuum = not lut.includes_co2_continuum
        backend = "lut"
    else:
        if parfile is not None:
            lines = read_hitran_par(parfile, record_format=record_format)
        optical_depth = build_optical_depth_stack(
            wavenumber,
            lines,
            layers,
            selected,
            line_mixing=line_mixing,
            voigt_method=voigt_method,
            tile_width_cm1=tile_width_cm1,
            line_cutoff_cm1=line_cutoff_cm1,
        )
        add_co2_continuum = True
        backend = "lbl"
    if include_continuum:
        optical_depth = optical_depth + build_continuum_optical_depth(
            wavenumber, layers, include_h2o=True, include_co2=add_co2_continuum
        )
    optical_depth = np.asarray(optical_depth, dtype=np.float64)
    invalid = ~np.isfinite(optical_depth) | (optical_depth < 0)
    if np.any(invalid):
        layer_index, frequency_index = np.argwhere(invalid)[0]
        raise ValueError(
            f"invalid total vertical optical depth at layer {layer_index}, "
            f"wavenumber {wavenumber[frequency_index]:g} cm^-1: "
            f"{optical_depth[layer_index, frequency_index]}"
        )
    upwelling, cumulative = integrate_upwelling(
        wavenumber,
        optical_depth,
        profile.temperature_k,
        mu_view=mu,
        pressure_hpa=profile.pressure_hpa,
        layer_mean=layer_mean,
    )
    downwelling = integrate_downwelling(
        wavenumber,
        optical_depth,
        profile.temperature_k,
        diffusivity=diffusivity,
        pressure_hpa=profile.pressure_hpa,
        layer_mean=layer_mean,
    )
    metadata = {
        "recipe": RECIPE_ID,
        "backend": backend,
        "diffusivity": float(diffusivity),
        "layer_mean": layer_mean,
        "include_continuum": bool(include_continuum),
        "gases": [gas.name for gas in selected],
        "isotopologues": {
            gas.name: [isotope.isotope_id for isotope in gas.isotopologues] for gas in selected
        },
        "requested_band_cm1": [low, high],
        "step_cm1": step,
    }
    if lut is not None:
        metadata["lut_provenance"] = dict(lut.provenance)
    else:
        available_pairs = lines.isotope_pairs
        metadata["line_cutoff_cm1"] = float(line_cutoff_cm1)
        metadata["voigt_method"] = voigt_method
        metadata["line_list_isotopologues"] = {
            gas.name: [
                isotope.isotope_id
                for isotope in gas.isotopologues
                if (gas.molecule_id, isotope.isotope_id) in available_pairs
            ]
            for gas in selected
        }
        metadata["absent_isotopologues"] = {
            gas.name: [
                isotope.isotope_id
                for isotope in gas.isotopologues
                if (gas.molecule_id, isotope.isotope_id) not in available_pairs
            ]
            for gas in selected
        }
        if parfile is not None:
            metadata["record_format"] = record_format
    return NadirForwardOutput(
        wavenumber,
        output_grid,
        upwelling,
        downwelling,
        cumulative,
        optical_depth,
        mu,
        ils,
        backend,
        metadata,
    )


def at_sensor_radiance_components(
    output: NadirForwardOutput, skin_temperature_k: float, epsilon
) -> dict[str, np.ndarray]:
    """Return surface/atmosphere/reflection and total on the internal pre-ILS grid.

    ``epsilon`` is a scalar or one value per internal wavenumber (including
    margin). No material lookup, implicit resampling or fallback emissivity.
    """
    skin_temperature = positive_scalar(skin_temperature_k, "skin_temperature_k")
    emissivity = np.asarray(epsilon, dtype=np.float64)
    if emissivity.ndim != 0 and emissivity.shape != output.wavenumber_cm1.shape:
        raise ValueError(
            "epsilon must be scalar or match output.wavenumber_cm1 including ILS margin"
        )
    if not np.all(np.isfinite(emissivity)) or np.any((emissivity < 0) | (emissivity > 1)):
        raise ValueError("epsilon must be finite and in [0, 1]")
    surface = (
        emissivity
        * planck_radiance_nu(output.wavenumber_cm1, skin_temperature)
        * output.total_transmittance
    )
    reflection = (1 - emissivity) * output.total_transmittance * output.downwelling_radiance
    return {
        "surface": surface,
        "atmosphere": output.upwelling_radiance,
        "reflection": reflection,
        "total": surface + output.upwelling_radiance + reflection,
    }


def at_sensor_radiance(
    output: NadirForwardOutput, skin_temperature_k: float, epsilon
) -> np.ndarray:
    """Return total radiance on output.output_wavenumber_cm1, optionally convolved."""
    radiance = at_sensor_radiance_components(output, skin_temperature_k, epsilon)["total"]
    if output.ils is None:
        return np.interp(output.output_wavenumber_cm1, output.wavenumber_cm1, radiance)
    return apply_ils_radiance(
        output.wavenumber_cm1,
        radiance,
        ils=output.ils,
        output_wavenumber_cm1=output.output_wavenumber_cm1,
    )
