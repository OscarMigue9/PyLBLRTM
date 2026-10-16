"""Independent reconstruction and storage/physics contract checks for SVD LUTs."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from photonix.lut import CrossSectionLUT


def _block(path, gas_name, start, end, *, offset=0.0, **changes):
    size = int(round((end - start) / 0.25)) + 1
    data = {
        "gas": gas_name,
        "a": start,
        "b": end,
        "NT": 2,
        "NP": 2,
        "T_LO": 200.0,
        "T_HI": 300.0,
        "P_LO": 0.01,
        "P_HI": 1.0,
        "dnu": 0.25,
        "r": 2,
        "U": np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "V": np.stack((np.linspace(0.2, 0.4, size), np.linspace(0.6, 0.8, size))).astype(
            np.float32
        ),
        "mu": (-24 + offset + np.arange(size) * 0.01).astype(np.float32),
    }
    data.update(changes)
    np.savez_compressed(path, **data)
    return data


@pytest.fixture
def fixture_lut(tmp_path):
    first = _block(tmp_path / "first.npz", "H2O", 1000.0, 1001.0)
    # Deliberately different overlap values expose accidental left/right ownership.
    second = _block(tmp_path / "second.npz", "H2O", 1001.0, 1002.0, offset=0.4)
    manifest = {
        "schema_version": 1,
        "storage_format": "svd_log10_cross_section",
        "recipe": "synthetic_fixture_only",
        "provenance": {"status": "unverified", "description": "analytic test factors"},
        "includes_h2o_continuum": False,
        "includes_co2_continuum": True,
        "gases": {"H2O": {"molecule_id": 1, "isotopologues": ["1", "2"], "vmr_ref": 0.01}},
        "blocks": [{"gas": "H2O", "file": "first.npz"}, {"gas": "H2O", "file": "second.npz"}],
        "self_broadening": {"method": "none"},
    }
    return tmp_path, manifest, (first, second)


def _dense_reference(block, grid, temperatures, pressures):
    """Reconstruct every state FIRST, then independently bilinearly interpolate."""
    logs = (
        block["mu"].astype(np.float64)
        + block["U"].astype(np.float64) @ block["V"].astype(np.float64)
    ).reshape(2, 2, -1)
    results = []
    for temperature, pressure in zip(temperatures, pressures):
        temperature_fraction = (temperature - 200.0) / 100.0
        pressure_fraction = (np.log10(pressure) + 2.0) / 2.0
        by_temperature = (1 - temperature_fraction) * logs[0] + temperature_fraction * logs[1]
        interpolated = (1 - pressure_fraction) * by_temperature[
            0
        ] + pressure_fraction * by_temperature[1]
        columns = np.rint((grid - block["a"]) / block["dnu"]).astype(int)
        results.append(10.0 ** interpolated[columns])
    return np.array(results)


def _species(name="H2O", molecule_id=1, isotopes=("1", "2")):
    return SimpleNamespace(
        name=name,
        molecule_id=molecule_id,
        isotopologues=tuple(SimpleNamespace(isotope_id=i) for i in isotopes),
    )


def _layer(temperature=250.0, pressure=0.1, column=2e20, vmr=0.01):
    return SimpleNamespace(
        temperature_k=temperature,
        pressure_atm=pressure,
        column_density_cm2={"H2O": column},
        vmr={"H2O": vmr},
    )


def test_interpolate_factors_equals_independent_dense_reference(fixture_lut):
    directory, manifest, blocks = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    grid = np.array([1000.0, 1000.5, 1001.0, 1001.75, 1002.0])
    temperatures = np.array([200.0, 230.0, 300.0])
    pressures = np.array([0.01, 0.2, 1.0])
    actual = lut.sigma("H2O", grid, temperatures, pressures)
    expected = np.concatenate(
        (
            _dense_reference(blocks[0], grid[:2], temperatures, pressures),
            _dense_reference(blocks[1], grid[2:], temperatures, pressures),
        ),
        axis=1,
    )
    # Two algebraically equivalent float64 reduction orders; exponentiation amplifies ulps.
    np.testing.assert_allclose(actual, expected, rtol=128 * np.finfo(float).eps, atol=0)
    assert actual.shape == (3, 5)
    assert actual.dtype == np.float64


def test_exact_final_endpoint_and_right_owned_overlap(fixture_lut):
    directory, manifest, blocks = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    for grid in (np.array([1001.0]), np.array([1002.0]), np.array([1001.0, 1002.0])):
        expected = _dense_reference(blocks[1], grid, [200.0], [0.01])
        np.testing.assert_array_equal(lut.sigma("H2O", grid, 200.0, 0.01), expected)


def test_equivalent_pressure_uses_exact_block_start_ratio(fixture_lut):
    directory, manifest, blocks = fixture_lut
    np.savez(directory / "ratios.npz", bordes=[999.0, 1000.0, 1001.0], H2O=[20.0, 2.0, 5.0])
    manifest["self_broadening"] = {"method": "effective_pressure", "ratios_file": "ratios.npz"}
    lut = CrossSectionLUT(directory, manifest=manifest)
    temperatures, pressures, vmr = np.array([250.0]), np.array([0.1]), np.array([0.2])
    grid = np.array([1000.5, 1001.0, 1002.0])
    expected = np.concatenate(
        [
            _dense_reference(blocks[0], grid[:1], temperatures, pressures * 1.2 / 1.01),
            _dense_reference(blocks[1], grid[1:], temperatures, pressures * 1.8 / 1.04),
        ],
        axis=1,
    )
    np.testing.assert_allclose(
        lut.sigma("H2O", grid, temperatures, pressures, vmr=vmr),
        expected,
        rtol=128 * np.finfo(float).eps,
        atol=0,
    )
    with pytest.raises(ValueError, match="vmr is required"):
        lut.sigma("H2O", grid, temperatures, pressures)


def test_out_of_domain_policy_is_explicit_and_applies_after_vmr_correction(fixture_lut):
    directory, manifest, blocks = fixture_lut
    np.savez(directory / "ratios.npz", bordes=[1000.0, 1001.0], H2O=[5.0, 5.0])
    manifest["self_broadening"] = {"method": "effective_pressure", "ratios_file": "ratios.npz"}
    lut = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(ValueError, match="effective pressure"):
        lut.sigma("H2O", [1000.0], [250.0], [0.8], vmr=[0.8])
    with pytest.raises(ValueError, match="outside LUT bounds"):
        lut.sigma("H2O", [1000.0], [150.0], [0.1], vmr=[0.01])
    clipped = CrossSectionLUT(directory, manifest=manifest, bounds="clip")
    expected = _dense_reference(blocks[0], np.array([1000.0]), [200.0], [1.0])
    np.testing.assert_array_equal(
        clipped.sigma("H2O", [1000.0], [150.0], [0.8], vmr=[0.8]), expected
    )


@pytest.mark.parametrize(
    "grid", [[999.75], [1002.25], [1000.0, 1000.0], [1001.0, 1000.0], [1000.125], [np.nan], []]
)
def test_spectral_coverage_order_and_node_alignment(fixture_lut, grid):
    directory, manifest, _ = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest, bounds="clip")
    with pytest.raises(ValueError):
        lut.sigma("H2O", grid, 250.0, 0.1)


def test_repeated_addition_grid_is_allowed(fixture_lut):
    directory, manifest, _ = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    grid = np.arange(1000.0, 1002.01, 0.25)
    assert lut.sigma("H2O", grid, 250.0, 0.1).shape == (1, grid.size)


@pytest.mark.parametrize(
    "temperatures,pressures,vmr",
    [
        ([250.0, 260.0], [0.1], None),
        ([[250.0]], [0.1], None),
        ([np.nan], [0.1], None),
        ([250.0], [0.0], None),
        ([250.0], [-1.0], None),
        ([250.0], [0.1], [1.01]),
        ([250.0], [0.1], [0.1, 0.2]),
    ],
)
def test_input_arrays_are_not_silently_broadcast(fixture_lut, temperatures, pressures, vmr):
    directory, manifest, _ = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(ValueError):
        lut.sigma("H2O", [1000.0], temperatures, pressures, vmr=vmr)


def test_optical_depth_units_isotope_contract_and_zero_column(fixture_lut):
    directory, manifest, _ = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    layers = [_layer(), _layer(column=0)]
    grid = [1000.0, 1002.0]
    expected = lut.sigma("H2O", grid, [250.0, 250.0], [0.1, 0.1]) * [[2e20], [0.0]]
    np.testing.assert_array_equal(lut.optical_depth(grid, layers, [_species()]), expected)
    with pytest.raises(ValueError, match="isotope mixture"):
        lut.optical_depth(grid, layers, [_species(isotopes=("1",))])
    with pytest.raises(ValueError, match="isotope mixture"):
        lut.optical_depth(grid, layers, [_species(molecule_id=2)])
    with pytest.raises(KeyError, match="absent"):
        lut.optical_depth(grid, layers, [_species("CO2", 2)])
    with pytest.raises(ValueError, match="duplicate"):
        lut.optical_depth(grid, layers, [_species(), _species()])
    with pytest.raises(ValueError, match="nonnegative"):
        lut.optical_depth(grid, [_layer(column=-1)], [_species()])
    missing = _layer()
    missing.vmr = {}
    with pytest.raises(ValueError, match="explicitly provide"):
        lut.optical_depth(grid, [missing], [_species()])


def test_lut_isotope_membership_allows_selected_gases(fixture_lut):
    directory, manifest, _ = fixture_lut
    manifest["gases"]["CO2"] = {"molecule_id": 2, "isotopologues": ["1"], "vmr_ref": 0.0004}
    for start in (1000.0, 1001.0):
        name = f"co2_{int(start)}.npz"
        _block(directory / name, "CO2", start, start + 1)
        manifest["blocks"].append({"gas": "CO2", "file": name})
    lut = CrossSectionLUT(directory, manifest=manifest)
    assert lut.gases == ("H2O", "CO2")
    assert lut.optical_depth([1000.0], [_layer()], [_species()]).shape == (1, 1)


def test_manifest_is_required_and_is_not_mutable_through_caller(fixture_lut):
    directory, manifest, _ = fixture_lut
    with pytest.raises(FileNotFoundError, match="explicit LUT manifest"):
        CrossSectionLUT(directory)
    (directory / "lut_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lut = CrossSectionLUT(directory)
    assert lut.recipe == manifest["recipe"]
    assert lut.includes_co2_continuum is True
    assert lut.includes_h2o_continuum is False
    provenance = lut.provenance
    provenance["status"] = "verified"
    assert lut.provenance["status"] == "unverified"
    lut = CrossSectionLUT(directory, manifest=manifest)
    manifest["gases"]["H2O"]["isotopologues"].clear()
    assert lut.optical_depth([1000.0], [_layer()], [_species()]).shape == (1, 1)


@pytest.mark.parametrize(
    "missing",
    [
        "recipe",
        "provenance",
        "includes_h2o_continuum",
        "includes_co2_continuum",
        "gases",
        "self_broadening",
    ],
)
def test_manifest_cannot_invent_missing_physics(fixture_lut, missing):
    directory, manifest, _ = fixture_lut
    del manifest[missing]
    with pytest.raises(ValueError):
        CrossSectionLUT(directory, manifest=manifest)


@pytest.mark.parametrize(
    "change",
    [
        {"V": np.ones((2, 4))},
        {"U": np.ones((3, 2))},
        {"mu": np.ones((1, 5))},
        {"P_LO": 0.0},
        {"dnu": 0.3},
        {"gas": "CO2"},
        {"r": 0},
        {"T_LO": 201.0},
        {"a": 1001.25, "b": 1002.25},
        {"a": 1000.75, "b": 1001.75},
    ],
)
def test_corrupt_shapes_axes_gas_and_coverage_fail_before_inference(fixture_lut, change):
    directory, manifest, _ = fixture_lut
    _block(directory / "second.npz", "H2O", 1001.0, 1002.0, **change)
    with pytest.raises((ValueError, TypeError)):
        CrossSectionLUT(directory, manifest=manifest)


def test_nonfinite_factors_and_overflow_are_errors(fixture_lut):
    directory, manifest, _ = fixture_lut
    _block(directory / "first.npz", "H2O", 1000.0, 1001.0, mu=np.full(5, np.nan))
    lut = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(ValueError, match="non-finite SVD"):
        lut.sigma("H2O", [1000.0], 250.0, 0.1)
    _block(directory / "first.npz", "H2O", 1000.0, 1001.0, mu=np.full(5, 400.0))
    lut = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(ValueError, match="non-finite reconstructed"):
        lut.sigma("H2O", [1000.0], 250.0, 0.1)


def test_bounded_immutable_cache_and_file_integrity(fixture_lut):
    directory, manifest, _ = fixture_lut
    manifest["blocks"][0]["sha256"] = hashlib.sha256(
        (directory / "first.npz").read_bytes()
    ).hexdigest()
    lut = CrossSectionLUT(directory, manifest=manifest, cache_blocks=1)
    lut.sigma("H2O", [1000.0], 250.0, 0.1)
    assert len(lut._cache) == 1
    assert all(not factor.flags.writeable for factor in next(iter(lut._cache.values())))
    lut.sigma("H2O", [1002.0], 250.0, 0.1)
    assert len(lut._cache) == 1
    assert next(iter(lut._cache)).name == "second.npz"
    no_cache = CrossSectionLUT(directory, manifest=manifest, cache_blocks=0)
    no_cache.sigma("H2O", [1000.0], 250.0, 0.1)
    assert not no_cache._cache
    manifest["blocks"][0]["sha256"] = "0" * 64
    bad_hash = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(ValueError, match="SHA256"):
        bad_hash.sigma("H2O", [1000.0], 250.0, 0.1)


def test_missing_gas_block_and_ratio_entry_are_explicit(fixture_lut):
    directory, manifest, _ = fixture_lut
    lut = CrossSectionLUT(directory, manifest=manifest)
    with pytest.raises(KeyError, match="absent"):
        lut.sigma("CO2", [1000.0], 250.0, 0.1)
    missing_block = deepcopy(manifest)
    missing_block["blocks"][1]["file"] = "absent.npz"
    with pytest.raises(FileNotFoundError):
        CrossSectionLUT(directory, manifest=missing_block)
    np.savez(directory / "ratios.npz", bordes=[999.0, 1000.0], H2O=[2.0, 3.0])
    manifest["self_broadening"] = {"method": "effective_pressure", "ratios_file": "ratios.npz"}
    with pytest.raises(ValueError, match="block start 1001"):
        CrossSectionLUT(directory, manifest=manifest)


def test_ratio_checksum_is_verified_when_declared(fixture_lut):
    directory, manifest, _ = fixture_lut
    path = directory / "ratios.npz"
    np.savez(path, bordes=[1000.0, 1001.0], H2O=[2.0, 3.0])
    manifest["self_broadening"] = {
        "method": "effective_pressure",
        "ratios_file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert CrossSectionLUT(directory, manifest=manifest).gases == ("H2O",)
    np.savez(path, bordes=[1000.0, 1001.0], H2O=[2.0, 4.0])
    with pytest.raises(ValueError, match="SHA256 mismatch for self-broadening"):
        CrossSectionLUT(directory, manifest=manifest)


@pytest.mark.skipif(
    not os.environ.get("PHOTONIX_LUT_TEST_DIRECTORY")
    or not os.environ.get("PHOTONIX_LUT_TEST_MANIFEST"),
    reason="real LUT smoke requires explicit external data and provenance manifest",
)
def test_optional_real_svd_block_without_copying_external_data():
    """Storage smoke only: this does not certify spectroscopy or historical provenance."""
    directory = Path(os.environ["PHOTONIX_LUT_TEST_DIRECTORY"])
    manifest = Path(os.environ["PHOTONIX_LUT_TEST_MANIFEST"])
    lut = CrossSectionLUT(directory, manifest=manifest, cache_blocks=1)
    gas = lut.gases[0]
    first = lut._blocks[gas][0]
    grid = [first.start_cm1, first.start_cm1 + lut._step, first.end_cm1]
    sigma = lut.sigma(
        gas,
        grid,
        [lut._temperature[0]],
        [10 ** lut._log_pressure[0]],
        vmr=[lut._gas_metadata[gas]["vmr_ref"]],
    )
    assert sigma.shape == (1, 3)
    assert np.all(np.isfinite(sigma)) and np.all(sigma >= 0)
