import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import wofz

from photonix.spectroscopy.voigt import voigt_profile


def test_scipy_matches_faddeeva_with_hwhm_convention():
    nu = np.linspace(-3, 3, 401)
    center = np.array([-0.2, 0.6])
    doppler = np.array([0.1, 0.03])
    lorentz = np.array([0.01, 0.5])
    factor = np.sqrt(np.log(2))
    z = factor * ((nu[:, None] - center) + 1j * lorentz) / doppler
    expected = factor / (np.sqrt(np.pi) * doppler) * wofz(z).real
    actual = voigt_profile(nu, center, doppler, lorentz, method="scipy")
    np.testing.assert_allclose(actual, expected, rtol=2e-15)


def test_humlicek_regions_match_scipy():
    # Covers all four W4 regions, their boundaries, and narrow Doppler cores.
    x = np.unique(np.r_[np.linspace(-20, 20, 1001), [-15, -5.5, 0, 5.5, 15]])
    y = np.array([0.0, 1e-6, 0.001, 0.1, 1.0, 5.5, 15.0, 100.0])
    doppler = np.full(y.size, np.sqrt(np.log(2)))
    exact = voigt_profile(x, np.zeros_like(y), doppler, y, method="scipy")
    actual = voigt_profile(x, np.zeros_like(y), doppler, y)
    np.testing.assert_allclose(actual, exact, rtol=1.0e-4, atol=1e-12)
    assert actual.dtype == np.float64


@pytest.mark.parametrize("method", ["humlicek", "scipy"])
def test_gaussian_area_and_symmetry(method):
    nu = np.linspace(-2, 2, 20001)
    profile = voigt_profile(nu, [0], [0.1], [0], method=method)[:, 0]
    np.testing.assert_allclose(profile, profile[::-1], atol=2e-13)
    # The W4 rational approximation is not an exactly normalised Gaussian.
    assert trapezoid(profile, nu) == pytest.approx(1, abs=1e-5 if method == "humlicek" else 1e-12)


@pytest.mark.parametrize("doppler,lorentz", [([0], [0.1]), ([0.1], [-1]), ([np.nan], [1])])
def test_invalid_widths_fail(doppler, lorentz):
    with pytest.raises(ValueError):
        voigt_profile([1000], [1000], doppler, lorentz)


def test_unknown_method_fails():
    with pytest.raises(ValueError, match="method"):
        voigt_profile([1000], [1000], [0.1], [0.1], method="legacy")
