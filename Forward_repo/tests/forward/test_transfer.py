import numpy as np
import pytest
from scipy.integrate import quad

from photonix.transfer.integrator import integrate_downwelling, integrate_upwelling
from photonix.transfer.planck import brightness_temperature_nu, planck_radiance_nu


def test_planck_round_trip_and_shape():
    grid = np.array([714.0, 1000.0, 1250.0])
    temperature = np.array([180.0, 250.0, 330.0])
    radiance = planck_radiance_nu(grid, temperature)
    np.testing.assert_allclose(
        brightness_temperature_nu(grid, radiance),
        np.broadcast_to(temperature[:, None], (3, 3)),
        rtol=2e-15,
    )
    assert planck_radiance_nu(grid, 300.0).shape == (3,)
    with pytest.raises(ValueError):
        brightness_temperature_nu(grid, [1, np.nan, 2])


@pytest.mark.parametrize("depth", [0.0, 1e-10, 1e-3, 0.1, 1.0, 20.0, 1000.0])
def test_isothermal_slab_exact_solution(depth):
    grid = np.array([750.0, 1000.0])
    optical_depth = np.full((1, 2), depth)
    planck = planck_radiance_nu(grid, 280.0)
    upwelling, transmittance = integrate_upwelling(grid, optical_depth, [280.0, 280.0], mu_view=0.5)
    downwelling = integrate_downwelling(grid, optical_depth, [280.0, 280.0])
    np.testing.assert_allclose(upwelling, planck * -np.expm1(-depth / 0.5), rtol=2e-14)
    np.testing.assert_allclose(downwelling, planck * -np.expm1(-depth * 1.66), rtol=2e-14)
    np.testing.assert_allclose(transmittance[-1], np.exp(-depth / 0.5))


@pytest.mark.parametrize("depth", [1e-5, 0.001, 0.1, 2.0, 25.0])
def test_temperature_gradient_against_independent_quadrature(depth):
    grid = np.array([900.0])
    top, bottom = planck_radiance_nu(grid, [210.0, 310.0])[:, 0]
    expected_up = quad(
        lambda x: (top + (bottom - top) * x) * depth * np.exp(-depth * x), 0.0, 1.0, epsabs=1e-12
    )[0]
    expected_down = quad(
        lambda x: (bottom + (top - bottom) * x) * depth * np.exp(-depth * x), 0.0, 1.0, epsabs=1e-12
    )[0]
    up, _ = integrate_upwelling(grid, [[depth]], [210.0, 310.0])
    down = integrate_downwelling(grid, [[depth]], [210.0, 310.0], diffusivity=1.0)
    np.testing.assert_allclose(up[0], expected_up, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(down[0], expected_down, rtol=2e-10, atol=1e-12)


def test_layer_subdivision_preserves_isothermal_column():
    grid = [850.0, 1000.0]
    whole = integrate_upwelling(grid, [[0.3, 2.0]], [260, 260])
    split = integrate_upwelling(grid, [[0.1, 0.5], [0.2, 1.5]], [260, 260, 260])
    np.testing.assert_allclose(whole[0], split[0], rtol=2e-15)
    np.testing.assert_allclose(whole[1][-1], split[1][-1], rtol=2e-15)


@pytest.mark.parametrize("mu", [0, -1, 1.1, np.nan, np.inf])
def test_bad_viewing_geometry_raises(mu):
    with pytest.raises(ValueError):
        integrate_upwelling([800.0], [[1.0]], [250.0, 280.0], mu_view=mu)


@pytest.mark.parametrize("depth", [-1.0, np.inf, np.nan])
def test_bad_optical_depth_raises(depth):
    with pytest.raises(ValueError):
        integrate_downwelling([800.0], [[depth]], [250.0, 280.0])


def test_layer_mean_is_explicit_not_selected_by_presence_of_pressure():
    args = ([850.0], [[1.0]], [200.0, 320.0])
    arithmetic = integrate_upwelling(*args)[0]
    with_pressure = integrate_upwelling(*args, pressure_hpa=[10, 1000])[0]
    density = integrate_upwelling(*args, pressure_hpa=[10, 1000], layer_mean="density")[0]
    np.testing.assert_array_equal(arithmetic, with_pressure)
    assert abs(float(density[0] - arithmetic[0])) > 1.0
