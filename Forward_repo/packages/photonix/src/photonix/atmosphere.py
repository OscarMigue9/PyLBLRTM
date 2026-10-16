"""Atmospheric state and hydrostatic layers, ordered from TOA to surface."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from photonix._validation import finite_vector, positive_scalar
from photonix.constants import (
    AVOGADRO_MOL_INV,
    DRY_AIR_MOLAR_MASS_KG_MOL,
    GRAVITY_M_S2,
    STANDARD_PRESSURE_HPA,
)

if TYPE_CHECKING:
    from photonix.spectroscopy.species import Species


@dataclass(frozen=True)
class Layer:
    """One layer: pressure [atm], temperature [K], VMR and columns [molecule/cm²]."""

    index: int
    pressure_atm: float
    temperature_k: float
    vmr: Mapping[str, float]
    column_density_cm2: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.index, (int, np.integer)) or self.index < 0:
            raise ValueError("index must be a nonnegative integer")
        object.__setattr__(self, "pressure_atm", positive_scalar(self.pressure_atm, "pressure_atm"))
        object.__setattr__(
            self, "temperature_k", positive_scalar(self.temperature_k, "temperature_k")
        )
        if set(self.vmr) != set(self.column_density_cm2):
            raise ValueError("vmr and column_density_cm2 must identify the same gases")
        fractions = {name: float(value) for name, value in self.vmr.items()}
        columns = {name: float(value) for name, value in self.column_density_cm2.items()}
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in fractions.values()):
            raise ValueError("layer vmr must be finite and between zero and one")
        if sum(fractions.values()) > 1 + 1e-12:
            raise ValueError("sum of layer vmr exceeds one")
        if any(not np.isfinite(v) or v < 0 for v in columns.values()):
            raise ValueError("column_density_cm2 must be finite and nonnegative")
        object.__setattr__(self, "vmr", MappingProxyType(fractions))
        object.__setattr__(self, "column_density_cm2", MappingProxyType(columns))


@dataclass(frozen=True)
class AtmosphericProfile:
    """Levels P [hPa], T [K], and gas VMR [mol/mol], ordered TOA first.

    Every gas has one VMR per level. Input arrays are copied and made read-only;
    layers use arithmetic temperature/VMR and geometric pressure means, matching
    the historical hydrostatic layer construction. Surface temperature and
    emissivity are separate inputs to the radiance calculation.
    """

    pressure_hpa: np.ndarray
    temperature_k: np.ndarray
    vmr: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        pressure = finite_vector(self.pressure_hpa, "pressure_hpa", minimum_size=2).copy()
        temperature = finite_vector(self.temperature_k, "temperature_k", minimum_size=2).copy()
        if pressure.shape != temperature.shape:
            raise ValueError("pressure_hpa and temperature_k must have equal shapes")
        if np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0):
            raise ValueError("pressure_hpa must be positive and strictly increase from TOA")
        if np.any(temperature <= 0):
            raise ValueError("temperature_k must be positive")
        fractions = {}
        total_fraction = np.zeros_like(pressure)
        for name, values in self.vmr.items():
            if not isinstance(name, str) or not name:
                raise ValueError("vmr gas names must be nonempty strings")
            fraction = finite_vector(values, f"vmr[{name}]").copy()
            if fraction.shape != pressure.shape or np.any((fraction < 0) | (fraction > 1)):
                raise ValueError(f"vmr[{name}] must have one value in [0, 1] per level")
            total_fraction += fraction
            fraction.setflags(write=False)
            fractions[name] = fraction
        if np.any(total_fraction > 1 + 1e-12):
            raise ValueError("sum of gas vmr exceeds one at a level")
        pressure.setflags(write=False)
        temperature.setflags(write=False)
        object.__setattr__(self, "pressure_hpa", pressure)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "vmr", MappingProxyType(fractions))

    @property
    def n_levels(self) -> int:
        return self.pressure_hpa.size

    @property
    def n_layers(self) -> int:
        return self.n_levels - 1

    def build_layers(self, species: Sequence[Species]) -> tuple[Layer, ...]:
        """Build vertical columns for selected gases; missing VMR is an error."""
        names = [gas.name for gas in species]
        if len(set(names)) != len(names):
            raise ValueError("species contains repeated gas names")
        missing = set(names) - self.vmr.keys()
        if missing:
            raise ValueError(f"profile is missing vmr for requested gases: {sorted(missing)}")
        column_factor = AVOGADRO_MOL_INV / (GRAVITY_M_S2 * DRY_AIR_MOLAR_MASS_KG_MOL)
        layers = []
        for index in range(self.n_layers):
            pressure_top, pressure_bottom = self.pressure_hpa[index : index + 2]
            air_column_cm2 = column_factor * (pressure_bottom - pressure_top) * 100.0 / 1e4
            fractions = {name: float(np.mean(self.vmr[name][index : index + 2])) for name in names}
            layers.append(
                Layer(
                    index=index,
                    pressure_atm=float(
                        np.sqrt(pressure_top * pressure_bottom) / STANDARD_PRESSURE_HPA
                    ),
                    temperature_k=float(np.mean(self.temperature_k[index : index + 2])),
                    vmr=fractions,
                    column_density_cm2={
                        name: fraction * air_column_cm2 for name, fraction in fractions.items()
                    },
                )
            )
        return tuple(layers)
