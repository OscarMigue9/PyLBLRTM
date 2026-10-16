"""H2O and CO2 MT_CKD continua of the thermal production recipe, loaded lazily."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np

from photonix.constants import SECOND_RADIATION_CONSTANT_CM_K, STANDARD_PRESSURE_HPA


@dataclass(frozen=True)
class ContinuumTables:
    """Reference coefficients. Runtime radiation constant follows production.

    H2O CS/CF use log-linear interpolation; its temperature exponent and CO2 use
    linear interpolation. Extrapolation is an error. The bundled source metadata's
    H2O C2=1.4388 is retained in its manifest, while production evaluates radiation
    with 1.4387752 cm K. H2O and CO2 reference pressure are 1013.0 hPa, distinct from
    the 1013.25 hPa conversion of atmospheric pressure to hPa.
    """

    h2o_wavenumber_cm1: np.ndarray
    h2o_self: np.ndarray
    h2o_foreign: np.ndarray
    h2o_temperature_exponent: np.ndarray
    co2_wavenumber_cm1: np.ndarray
    co2_coefficient: np.ndarray
    h2o_reference_temperature_k: float = 296.0
    h2o_reference_pressure_hpa: float = 1013.0
    co2_reference_temperature_k: float = 296.0
    co2_reference_pressure_hpa: float = 1013.0

    def __post_init__(self):
        names = (
            "h2o_wavenumber_cm1",
            "h2o_self",
            "h2o_foreign",
            "h2o_temperature_exponent",
            "co2_wavenumber_cm1",
            "co2_coefficient",
        )
        for name in names:
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.ndim != 1 or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite one-dimensional array")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        for grid in (self.h2o_wavenumber_cm1, self.co2_wavenumber_cm1):
            if grid.size < 2 or np.any(np.diff(grid) <= 0):
                raise ValueError("Continuum reference grids must increase strictly")
        if any(
            getattr(self, name).shape != self.h2o_wavenumber_cm1.shape
            for name in ("h2o_self", "h2o_foreign", "h2o_temperature_exponent")
        ):
            raise ValueError("H2O continuum coefficients must align with their frequency grid")
        if self.co2_coefficient.shape != self.co2_wavenumber_cm1.shape:
            raise ValueError("CO2 continuum coefficients must align with their frequency grid")
        if np.any(self.h2o_self <= 0) or np.any(self.h2o_foreign <= 0):
            raise ValueError("Log-interpolated H2O coefficients must be strictly positive")
        if np.any(self.co2_coefficient < 0):
            raise ValueError("CO2 continuum coefficients must be nonnegative")
        for name in (
            "h2o_reference_temperature_k",
            "h2o_reference_pressure_hpa",
            "co2_reference_temperature_k",
            "co2_reference_pressure_hpa",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ContinuumTables":
        """Load only numerical H2O/CO2 tables; never pickle or import-time I/O."""
        resource = (
            Path(path)
            if path is not None
            else files("photonix").joinpath("resources/aer/continuum.npz")
        )
        with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            return cls(
                h2o_wavenumber_cm1=data["h2o_NU_TABLE"],
                h2o_self=data["h2o_CS_REF"],
                h2o_foreign=data["h2o_CF_REF"],
                h2o_temperature_exponent=data["h2o_TEXP_REF"],
                co2_wavenumber_cm1=data["co2_nu"],
                co2_coefficient=data["co2_S"],
                h2o_reference_temperature_k=float(data["h2o_T_REF"]),
                h2o_reference_pressure_hpa=float(data["h2o_P_REF_HP"]),
            )


def _radiation_term(wavenumber_cm1, temperature_k):
    u = wavenumber_cm1 * (SECOND_RADIATION_CONSTANT_CM_K / temperature_k)
    result = np.empty_like(u)
    low, high = u <= 0.01, u > 10.0
    middle = ~(low | high)
    result[low] = 0.5 * u[low] * wavenumber_cm1[low]
    result[high] = wavenumber_cm1[high]
    result[middle] = wavenumber_cm1[middle] * np.tanh(0.5 * u[middle])
    return result


def _check_coverage(nu, reference, gas):
    if nu.min() < reference[0] or nu.max() > reference[-1]:
        raise ValueError(
            f"{gas} continuum tables cover {reference[0]}..{reference[-1]} cm^-1; "
            f"requested {nu.min()}..{nu.max()} cm^-1"
        )


def build_continuum_optical_depth(
    wavenumber_cm1, layers, *, tables=None, include_h2o=True, include_co2=True
) -> np.ndarray:
    """Return vertical continuum OD [layer, frequency], adding each gas once."""
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    if nu.ndim != 1 or nu.size == 0 or not np.isfinite(nu).all() or np.any(nu <= 0):
        raise ValueError("Continuum frequencies must be finite, positive and one-dimensional")
    layers = tuple(layers)
    result = np.zeros((len(layers), nu.size), dtype=np.float64)
    if not include_h2o and not include_co2:
        return result
    tables = ContinuumTables.load() if tables is None else tables
    if include_h2o:
        _check_coverage(nu, tables.h2o_wavenumber_cm1, "H2O")
        self_coefficient = np.exp(np.interp(nu, tables.h2o_wavenumber_cm1, np.log(tables.h2o_self)))
        foreign_coefficient = np.exp(
            np.interp(nu, tables.h2o_wavenumber_cm1, np.log(tables.h2o_foreign))
        )
        temperature_exponent = np.interp(
            nu, tables.h2o_wavenumber_cm1, tables.h2o_temperature_exponent
        )
    if include_co2:
        _check_coverage(nu, tables.co2_wavenumber_cm1, "CO2")
        co2_coefficient = np.interp(nu, tables.co2_wavenumber_cm1, tables.co2_coefficient)
    for k, layer in enumerate(layers):
        temperature, pressure = float(layer.temperature_k), float(layer.pressure_atm)
        if (
            not np.isfinite(temperature)
            or temperature <= 0
            or not np.isfinite(pressure)
            or pressure <= 0
        ):
            raise ValueError(f"Layer {layer.index}: continuum requires positive finite T and P")
        radiation = _radiation_term(nu, temperature)
        for gas, enabled in (("H2O", include_h2o), ("CO2", include_co2)):
            if not enabled:
                continue
            column = float(layer.column_density_cm2.get(gas, 0.0))
            if not np.isfinite(column) or column < 0:
                raise ValueError(f"Layer {layer.index}, {gas}: invalid continuum column density")
            if column == 0:
                continue
            if gas == "H2O":
                if gas not in layer.vmr:
                    raise ValueError(f"Layer {layer.index}: H2O continuum requires H2O VMR")
                vmr = float(layer.vmr[gas])
                if not np.isfinite(vmr) or not 0 <= vmr <= 1:
                    raise ValueError(f"Layer {layer.index}: H2O VMR must be in [0, 1]")
                temperature_ratio = tables.h2o_reference_temperature_k / temperature
                density_ratio = (
                    pressure * STANDARD_PRESSURE_HPA / tables.h2o_reference_pressure_hpa
                ) * temperature_ratio
                result[k] += (
                    (
                        self_coefficient * temperature_ratio**temperature_exponent * vmr
                        + foreign_coefficient * (1.0 - vmr)
                    )
                    * density_ratio
                    * radiation
                    * column
                )
            else:
                density_ratio = (
                    pressure
                    * STANDARD_PRESSURE_HPA
                    / tables.co2_reference_pressure_hpa
                    * tables.co2_reference_temperature_k
                    / temperature
                )
                result[k] += co2_coefficient * column * density_ratio * 1.0e-20 * radiation
        if not np.isfinite(result[k]).all() or np.any(result[k] < 0):
            raise FloatingPointError(f"Layer {layer.index}: invalid continuum optical depth")
    return result
