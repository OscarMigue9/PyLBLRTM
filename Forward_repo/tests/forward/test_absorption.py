from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import wofz

from photonix.atmosphere import Layer
from photonix.constants import (
    DOPPLER_COEFFICIENT,
    REFERENCE_TEMPERATURE_K,
    SECOND_RADIATION_CONSTANT_CM_K,
)
from photonix.spectroscopy.absorption import build_optical_depth_stack
from photonix.spectroscopy.aer import AerLineMixing, aer_line_profile, coupling_at_temperature
from photonix.spectroscopy.hitran import LineList
from photonix.spectroscopy.species import Isotopologue, Species


def _lines(molecule=1, center=1000.0, shift=0.0, width=0.07, strength=1e-22):
    return LineList(
        np.array([molecule]),
        np.array(["1"]),
        np.array([center]),
        np.array([strength]),
        np.array([0.0]),
        np.array([width]),
        np.array([width]),
        np.array([0.0]),
        np.array([0.7]),
        np.array([shift]),
    )


def _species(tmp_path, name="H2O", molecule=1):
    path = tmp_path / "partition.txt"
    path.write_text("100 1\n296 1\n400 1\n", encoding="ascii")
    return (Species(name, molecule, (Isotopologue("1", 44.0 if molecule == 2 else 18.0, path),)),)


def _layer(name="H2O", pressure=1.0, column=1e20, temperature=296.0):
    return Layer(0, pressure, temperature, {name: 0.01}, {name: column})


def test_partitioned_tiles_equal_single_tile_including_both_endpoints(tmp_path):
    nu = np.linspace(997, 1003, 601)
    lines, species, layers = _lines(), _species(tmp_path), [_layer()]
    single = build_optical_depth_stack(nu, lines, layers, species, tile_width_cm1=20)
    tiled = build_optical_depth_stack(nu, lines, layers, species, tile_width_cm1=2)
    np.testing.assert_array_equal(tiled, single)
    assert single[0, -1] > 0


def test_direct_optical_depth_has_no_transmittance_underflow_ceiling(tmp_path):
    nu = np.linspace(999.9, 1000.1, 21)
    species = _species(tmp_path)
    small = build_optical_depth_stack(nu, _lines(), [_layer()], species)
    large = build_optical_depth_stack(nu, _lines(), [_layer(column=1e27)], species)
    np.testing.assert_allclose(large, small * 1e7, rtol=2e-15)
    assert large.max() > 1e5
    assert np.isfinite(large).all()


def test_isotope_selection_adds_each_abundance_weighted_strength_once(tmp_path):
    nu = np.linspace(999.5, 1000.5, 101)
    original = _species(tmp_path)[0]
    second = Isotopologue("2", 20.0, original.isotopologues[0].partition_file)
    both = Species("H2O", 1, original.isotopologues + (second,))
    first_lines = _lines()
    arrays = {
        name: np.repeat(getattr(first_lines, name), 2)
        for name in (
            "molecule_id",
            "isotope_id",
            "wavenumber_cm1",
            "line_strength_ref",
            "einstein_a_s",
            "air_width_cm1_atm",
            "self_width_cm1_atm",
            "lower_energy_cm1",
            "temperature_exponent",
            "pressure_shift_cm1_atm",
        )
    }
    arrays["isotope_id"] = np.array(["1", "2"])
    arrays["wavenumber_cm1"] = np.array([999.9, 1000.1])
    arrays["line_strength_ref"] = np.array([1e-22, 2e-24])
    lines = LineList(**arrays)
    total = build_optical_depth_stack(nu, lines, [_layer()], [both])
    first = build_optical_depth_stack(nu, lines, [_layer()], [original])
    second_only = build_optical_depth_stack(nu, lines, [_layer()], [Species("H2O", 1, (second,))])
    np.testing.assert_allclose(total, first + second_only, rtol=2e-15)


def test_pressure_shift_is_computed_per_layer(tmp_path):
    nu = np.linspace(999.9, 1000.4, 501)
    layers = [_layer(pressure=0.2), _layer(pressure=1.0)]
    lines = _lines(shift=0.3, width=0.0)
    result = build_optical_depth_stack(nu, lines, layers, _species(tmp_path), voigt_method="scipy")
    centers = nu[np.argmax(result, axis=1)]
    np.testing.assert_allclose(
        centers, [1000 + 0.3 * 0.2 * 0.99, 1000 + 0.3 * 1.0 * 0.99], atol=0.00051
    )


def test_radiation_term_is_applied_at_each_frequency(tmp_path):
    # LBLRTM JRAD=0 / SpectralCalc Eq. 11: k = S(T) f(nu) R(nu, T) / R(nu_c, T).
    nu = np.linspace(976.0, 1024.0, 481)
    temperature, column = 250.0, 1e20
    layer = _layer(temperature=temperature, column=column)
    species = _species(tmp_path)
    actual = build_optical_depth_stack(nu, _lines(), [layer], species, voigt_method="scipy")[0]
    c2 = SECOND_RADIATION_CONSTANT_CM_K
    lorentz = 0.07 * (REFERENCE_TEMPERATURE_K / temperature) ** 0.7
    doppler = 1000.0 * DOPPLER_COEFFICIENT * np.sqrt(temperature / 18.0)
    profile = aer_line_profile(nu, [1000.0], [doppler], [lorentz], voigt_method="scipy")[:, 0]
    strength = (
        1e-22
        * (1 - np.exp(-c2 * 1000.0 / temperature))
        / (1 - np.exp(-c2 * 1000.0 / REFERENCE_TEMPERATURE_K))
        * column
    )

    def radiation(v):
        return v * np.tanh(0.5 * c2 * v / temperature)

    expected = strength * profile * radiation(nu) / radiation(1000.0)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    # 24 cm-1 below the centre the wing is ~2.5 % weaker than the line-centre convention.
    assert actual[0] / (strength * profile[0]) < 0.98


def test_aer_co2_multiplicative_shape_matches_independent_faddeeva():
    nu = np.array([973.0, 975.0, 990.0, 1000.0, 1007.0, 1025.0, 1027.0])
    center, doppler, lorentz = np.array([1000.0]), np.array([0.003]), np.array([0.07])
    coupling, pressure, cutoff = np.array([0.012]), 0.8, 25.0
    actual = aer_line_profile(
        nu,
        center,
        doppler,
        lorentz,
        is_co2=True,
        coupling=coupling,
        pressure_atm=pressure,
        voigt_method="scipy",
    )[:, 0]
    d = nu - center[0]
    s2 = np.sqrt(np.log(2))
    v = wofz(s2 * (d + 1j * lorentz[0]) / doppler[0]).real * s2 / np.sqrt(np.pi) / doppler[0]
    pedestal = lorentz[0] / np.pi / (cutoff**2 + lorentz[0] ** 2)
    alpha_v = 0.5346 * lorentz[0] + np.sqrt(0.2166 * lorentz[0] ** 2 + doppler[0] ** 2)
    expected = (v - (2 - d**2 / cutoff**2) * pedestal) * (1 + d * coupling[0] * pressure / alpha_v)
    expected[np.abs(d) > cutoff] = 0
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-17)


def test_nonco2_has_constant_pedestal_and_strict_line_cutoff():
    nu = np.array([974.99, 975.0, 990.0, 1000.0, 1025.0, 1025.01])
    actual = aer_line_profile(nu, [1000], [0.003], [0.07], voigt_method="scipy")[:, 0]
    d = nu - 1000
    s2 = np.sqrt(np.log(2))
    expected = wofz(s2 * (d + 0.07j) / 0.003).real * s2 / np.sqrt(np.pi) / 0.003
    expected -= 0.07 / np.pi / (25**2 + 0.07**2)
    expected[np.abs(d) > 25] = 0
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-17)


def test_co2_requires_explicit_matched_coupling(tmp_path):
    nu, lines = np.linspace(999, 1001, 101), _lines(molecule=2)
    species = _species(tmp_path, "CO2", 2)
    with pytest.raises(ValueError, match="explicit"):
        build_optical_depth_stack(nu, lines, [_layer("CO2")], species)
    coupling = AerLineMixing(np.array(["1"]), np.array([1000.0]), np.zeros((1, 4)))
    result = build_optical_depth_stack(nu, lines, [_layer("CO2")], species, line_mixing=coupling)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="Missing"):
        coupling.coefficients_for(_lines(molecule=2, center=1001.0))


def test_coupling_matches_isotope_nearest_frequency_and_singleton():
    coupling = AerLineMixing(
        np.array(["1", "2"]), np.array([1000.0, 1000.0]), np.array([[1, 2, 3, 4], [5, 6, 7, 8.0]])
    )
    lines = SimpleNamespace(
        molecule_id=np.array([2, 2]),
        isotope_id=np.array(["2", "1"]),
        wavenumber_cm1=np.array([1000.00001, 1000.0]),
    )
    np.testing.assert_array_equal(coupling.coefficients_for(lines), [[5, 6, 7, 8], [1, 2, 3, 4]])
    ambiguous = AerLineMixing(
        np.array(["1", "1"]), np.array([1000.0, 1000.0]), np.array([[1, 2, 3, 4], [5, 6, 7, 8.0]])
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        ambiguous.coefficients_for(_lines(molecule=2))


def test_coupling_reader_keeps_explicit_zero_and_parses_d_exponents(tmp_path):
    def header(isotope, frequency, flag):
        return f" 2{isotope}{frequency:12.6f}".ljust(98) + f"{flag:2d}"

    coefficient_row = (
        " 2" + "".join(f"{x:12.4E}" for x in [1.0, 0.0, -2.0, 0.0, 3.0, 0.0, -4.0, 0.0]) + " 1"
    )
    path = tmp_path / "lncpl_lines"
    path.write_text(
        "> AER copyright © header\n"
        + header("1", 1000.0, 1)
        + "\n"
        + coefficient_row.replace("E", "D")
        + "\n"
        + header("2", 1001.0, 0)
        + "\n",
        encoding="utf-8",
    )
    result = AerLineMixing.load(path)
    np.testing.assert_array_equal(result.coefficients, [[1, -2, 3, -4], [0, 0, 0, 0]])
    path.write_text(header("1", 1000.0, 1) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="continuation"):
        AerLineMixing.load(path)


def test_coupling_temperature_endpoints_are_explicit_production_convention():
    coefficients = np.array([[1, 2, 3, 4.0]])
    assert coupling_at_temperature(coefficients, 180)[0] == 1
    assert coupling_at_temperature(coefficients, 400)[0] == 4
    assert coupling_at_temperature(coefficients, 273)[0] == 2.5


def test_invalid_tiles_and_grid_fail_before_spectroscopy(tmp_path):
    species = _species(tmp_path)
    for tile in (0, 0.01, np.nan):
        with pytest.raises(ValueError, match="tile_width"):
            build_optical_depth_stack(
                [999.0, 999.01], _lines(), [_layer()], species, tile_width_cm1=tile
            )
    with pytest.raises(ValueError, match="uniform"):
        build_optical_depth_stack([999.0, 999.01, 999.03], _lines(), [_layer()], species)
    with pytest.raises(ValueError, match="finite"):
        build_optical_depth_stack([999.0, np.nan], _lines(), [_layer()], species)


def test_zero_column_and_empty_species_give_zero(tmp_path):
    nu = np.linspace(999, 1001, 11)
    output = build_optical_depth_stack(nu, _lines(), [_layer(column=0)], _species(tmp_path))
    np.testing.assert_array_equal(output, np.zeros((1, len(nu))))
    np.testing.assert_array_equal(build_optical_depth_stack(nu, _lines(), [_layer()], []), output)


def test_input_linelists_reject_nan_instead_of_transparency():
    with pytest.raises(ValueError, match="nonfinite"):
        replace(_lines(), line_strength_ref=np.array([np.nan]))
