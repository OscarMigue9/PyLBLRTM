import hashlib
from importlib.resources import files
import json

import numpy as np
import pytest

from photonix.atmosphere import Layer
from photonix.spectroscopy.continuum import ContinuumTables, build_continuum_optical_depth


def _layer(temperature=260.0, water=0.012, co2=0.0004):
    return Layer(
        0, 0.8, temperature, {"H2O": water, "CO2": co2}, {"H2O": 4e21 * water, "CO2": 4e21 * co2}
    )


def test_bundled_provenance_hashes_and_no_unrelated_tables():
    folder = files("photonix").joinpath("resources/aer")
    manifest = json.loads(folder.joinpath("manifest.json").read_text(encoding="utf-8"))
    payload = folder.joinpath("continuum.npz").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["bundle_sha256"]
    with folder.joinpath("continuum.npz").open("rb") as handle, np.load(handle) as data:
        assert set(data.files) == set(manifest["arrays"])
        for key in data.files:
            assert key.startswith(("h2o_", "co2_"))
            digest = hashlib.sha256(np.asarray(data[key], dtype="<f8").tobytes()).hexdigest()
            assert digest == manifest["arrays"][key]["sha256_float64_c_order"]


def test_continuum_matches_frozen_production_formula():
    tables = ContinuumTables.load()
    nu = np.array([712.5, 714, 831.3, 1000, 1249.9, 1251.5])
    layer = _layer()
    actual = build_continuum_optical_depth(nu, [layer], tables=tables)[0]
    temperature, pressure = layer.temperature_k, layer.pressure_atm
    u = nu * 1.4387752 / temperature
    radiation = np.where(
        u <= 0.01, 0.5 * u * nu, np.where(u <= 10, nu * (1 - np.exp(-u)) / (1 + np.exp(-u)), nu)
    )
    cs = np.exp(np.interp(nu, tables.h2o_wavenumber_cm1, np.log(tables.h2o_self)))
    cf = np.exp(np.interp(nu, tables.h2o_wavenumber_cm1, np.log(tables.h2o_foreign)))
    tx = np.interp(nu, tables.h2o_wavenumber_cm1, tables.h2o_temperature_exponent)
    co2 = np.interp(nu, tables.co2_wavenumber_cm1, tables.co2_coefficient)
    rho = pressure * 1013.25 / 1013.0 * 296.0 / temperature
    water = layer.vmr["H2O"]
    expected = (
        (cs * (296.0 / temperature) ** tx * water + cf * (1 - water))
        * rho
        * radiation
        * layer.column_density_cm2["H2O"]
    )
    expected += co2 * layer.column_density_cm2["CO2"] * rho * 1e-20 * radiation
    np.testing.assert_allclose(actual, expected, rtol=3e-15)


def test_continua_are_separable_and_disabled_is_zero():
    nu = np.linspace(714, 1250, 31)
    layer = _layer()
    water = build_continuum_optical_depth(nu, [layer], include_co2=False)
    carbon = build_continuum_optical_depth(nu, [layer], include_h2o=False)
    total = build_continuum_optical_depth(nu, [layer])
    np.testing.assert_array_equal(total, water + carbon)
    np.testing.assert_array_equal(
        build_continuum_optical_depth(nu, [layer], include_h2o=False, include_co2=False),
        np.zeros((1, len(nu))),
    )


def test_water_continuum_is_not_linear_in_water_abundance():
    nu = np.array([900.0, 1000.0])
    original = build_continuum_optical_depth(nu, [_layer(water=0.01)], include_co2=False)
    doubled = build_continuum_optical_depth(nu, [_layer(water=0.02)], include_co2=False)
    assert np.all(doubled > 2 * original)


def test_no_flat_extrapolation_or_nonfinite_input():
    with pytest.raises(ValueError, match="cover"):
        build_continuum_optical_depth([619.9, 714], [_layer()])
    with pytest.raises(ValueError, match="finite"):
        build_continuum_optical_depth([714, np.nan], [_layer()])


def test_interpolation_is_logarithmic_for_water_coefficients():
    tables = ContinuumTables(
        np.array([900.0, 1000.0]),
        np.array([1e-26, 1e-24]),
        np.array([1e-29, 1e-27]),
        np.array([0.0, 0.0]),
        np.array([900.0, 1000.0]),
        np.zeros(2),
    )
    layer = _layer(temperature=296, water=1, co2=0)
    actual = build_continuum_optical_depth([950], [layer], tables=tables)[0, 0]
    radiation = 950 * np.tanh(950 * 1.4387752 / 296 / 2)
    expected = 1e-25 * (0.8 * 1013.25 / 1013.0) * radiation * 4e21
    assert actual == pytest.approx(expected, rel=2e-14)
