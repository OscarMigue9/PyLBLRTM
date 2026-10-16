"""Checks for the raw catalog transformations used before the LUT."""

from pathlib import Path

import numpy as np

from pipelines.generation.l137 import full_level_pressures
from pipelines.generation.raw import _read_spectrum


def test_l137_pressure_grid_has_top_to_surface_order() -> None:
    pressures = full_level_pressures(np.array([101325.0, 90000.0]))
    assert pressures.shape == (2, 137)
    assert np.all(np.diff(pressures, axis=1) > 0)
    assert pressures[0, -1] > pressures[1, -1]


def test_ecostress_parser_sorts_wavelength_and_converts_percent(tmp_path: Path) -> None:
    spectrum = tmp_path / "sample.spectrum.txt"
    spectrum.write_text("header\n13.0 30\n8.0 10\n10.0 20\n", encoding="utf-8")
    wavelength, reflectance = _read_spectrum(spectrum)
    np.testing.assert_array_equal(wavelength, [8.0, 10.0, 13.0])
    np.testing.assert_allclose(reflectance, [0.1, 0.2, 0.3])
