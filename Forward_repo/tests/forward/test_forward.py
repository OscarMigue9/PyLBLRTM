import numpy as np
import pytest

from photonix import (
    AtmosphericProfile,
    InstrumentLineShape,
    LineList,
    at_sensor_radiance,
    at_sensor_radiance_components,
    default_species,
    planck_radiance_nu,
    run_nadir_simulation,
)


@pytest.fixture
def water_line():
    return LineList(
        molecule_id=[1],
        isotope_id=["1"],
        wavenumber_cm1=[900.0],
        line_strength_ref=[1e-22],
        einstein_a_s=[1.0],
        air_width_cm1_atm=[0.07],
        self_width_cm1_atm=[0.3],
        lower_energy_cm1=[100.0],
        temperature_exponent=[0.7],
        pressure_shift_cm1_atm=[-0.001],
    )


@pytest.fixture
def water_species():
    return default_species(["H2O"], isotopologues=[(1, "1")])


def test_transparent_black_surface_and_ils_edges(water_line, water_species):
    profile = AtmosphericProfile([100.0, 1000.0], [220.0, 280.0], {"H2O": [0.0, 0.0]})
    output = run_nadir_simulation(
        profile,
        water_species,
        lines=water_line,
        nu_min_cm1=899.0,
        nu_max_cm1=901.0,
        step_cm1=0.01,
        ils=InstrumentLineShape(),
    )
    np.testing.assert_array_equal(output.optical_depth_vertical, 0.0)
    np.testing.assert_array_equal(output.total_transmittance, 1.0)
    radiance = at_sensor_radiance(output, 290.0, 1.0)
    expected = planck_radiance_nu(output.output_wavenumber_cm1, 290.0)
    # Planck has curvature, so a finite-width Gaussian is not identical to its
    # unsmoothed value; its small convolution effect must not be an edge loss.
    np.testing.assert_allclose(radiance, expected, rtol=2e-6)
    assert radiance.shape == output.output_wavenumber_cm1.shape
    assert output.wavenumber_cm1[0] < 899.0
    assert output.wavenumber_cm1[-1] > 901.0


def test_view_angle_changes_upwelling_but_not_vertical_depth_or_downwelling(
    water_line, water_species
):
    profile = AtmosphericProfile(
        [10.0, 100.0, 1000.0], [220.0, 260.0, 280.0], {"H2O": [1e-5, 1e-3, 0.01]}
    )
    kwargs = dict(lines=water_line, nu_min_cm1=899.5, nu_max_cm1=900.5, step_cm1=0.01)
    nadir = run_nadir_simulation(profile, water_species, **kwargs)
    oblique = run_nadir_simulation(profile, water_species, mu_view=0.5, **kwargs)
    np.testing.assert_array_equal(nadir.optical_depth_vertical, oblique.optical_depth_vertical)
    np.testing.assert_array_equal(nadir.downwelling_radiance, oblique.downwelling_radiance)
    np.testing.assert_allclose(
        oblique.total_transmittance, nadir.total_transmittance**2, atol=2e-15
    )
    assert np.max(np.abs(nadir.upwelling_radiance - oblique.upwelling_radiance)) > 0.0


def test_components_sum_and_epsilon_contract(water_line, water_species):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.005, 0.01]})
    output = run_nadir_simulation(
        profile, water_species, lines=water_line, nu_min_cm1=899.0, nu_max_cm1=901.0, step_cm1=0.01
    )
    parts = at_sensor_radiance_components(output, 300.0, 0.9)
    np.testing.assert_array_equal(
        parts["total"], parts["surface"] + parts["atmosphere"] + parts["reflection"]
    )
    np.testing.assert_allclose(at_sensor_radiance(output, 300.0, 0.9), parts["total"])
    for bad in (-0.1, 1.1, np.nan, [0.9, 0.9]):
        with pytest.raises(ValueError):
            at_sensor_radiance(output, 300.0, bad)


def test_continuum_addition_is_exactly_once(water_line, water_species):
    from photonix.spectroscopy.continuum import build_continuum_optical_depth

    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.005, 0.01]})
    kwargs = dict(lines=water_line, nu_min_cm1=899.0, nu_max_cm1=901.0, step_cm1=0.01)
    only_lines = run_nadir_simulation(profile, water_species, include_continuum=False, **kwargs)
    complete = run_nadir_simulation(profile, water_species, **kwargs)
    continuum = build_continuum_optical_depth(
        complete.wavenumber_cm1, profile.build_layers(water_species)
    )
    np.testing.assert_allclose(
        complete.optical_depth_vertical - only_lines.optical_depth_vertical,
        continuum,
        rtol=1e-12,
        atol=1e-18,
    )


def test_backend_and_band_must_be_explicit(water_line, water_species):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.0, 0.0]})
    with pytest.raises(ValueError, match="exactly one"):
        run_nadir_simulation(profile, water_species)
    with pytest.raises(ValueError, match="714"):
        run_nadir_simulation(profile, water_species, lines=water_line, nu_min_cm1=645.0)


def test_missing_active_gas_fails_instead_of_becoming_transparent(water_line):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"CO2": [0.0004, 0.0004]})
    with pytest.raises(ValueError, match="CO2: no selected isotopologue"):
        run_nadir_simulation(
            profile,
            lines=water_line,
            include_continuum=False,
            nu_min_cm1=899.0,
            nu_max_cm1=901.0,
            step_cm1=0.01,
        )


def test_partial_line_list_coverage_is_visible_in_warning_and_metadata(water_line):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.001, 0.01]})
    with pytest.warns(RuntimeWarning, match="absent from the input"):
        output = run_nadir_simulation(
            profile, lines=water_line, nu_min_cm1=899.0, nu_max_cm1=901.0, step_cm1=0.01
        )
    assert output.metadata["line_list_isotopologues"] == {"H2O": ["1"]}
    assert output.metadata["absent_isotopologues"] == {"H2O": list("234567")}


def test_no_lines_in_window_is_distinct_from_missing_isotope(water_line, water_species):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.001, 0.01]})
    output = run_nadir_simulation(
        profile,
        water_species,
        lines=water_line,
        include_continuum=False,
        nu_min_cm1=1000.0,
        nu_max_cm1=1001.0,
        step_cm1=0.01,
    )
    np.testing.assert_array_equal(output.optical_depth_vertical, 0.0)
    assert output.metadata["absent_isotopologues"] == {"H2O": []}


def test_lut_rejects_line_calculation_options_before_using_stored_spectroscopy(
    water_line, water_species
):
    profile = AtmosphericProfile([100.0, 1000.0], [240.0, 290.0], {"H2O": [0.001, 0.01]})
    with pytest.raises(ValueError, match="LBL-only"):
        run_nadir_simulation(profile, water_species, lut=object(), voigt_method="scipy")
