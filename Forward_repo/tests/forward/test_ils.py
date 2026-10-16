import numpy as np
import pytest

from photonix.transfer.ils import InstrumentLineShape, apply_ils_radiance


def test_constant_radiance_preserved_at_requested_edges():
    ils = InstrumentLineShape(fwhm_cm1=1.0)
    step = 0.01
    padding = ils.half_width_samples(step)
    grid = 750 + np.arange(-padding, 201 + padding) * step
    output = np.arange(750.0, 752.01, 0.25)
    values = apply_ils_radiance(grid, np.ones_like(grid), ils=ils, output_wavenumber_cm1=output)
    np.testing.assert_allclose(values, 1.0, atol=3e-15)


def test_ils_refuses_missing_spectral_neighbors():
    grid = np.arange(750.0, 752.0, 0.01)
    with pytest.raises(ValueError, match="padding"):
        apply_ils_radiance(
            grid,
            np.ones_like(grid),
            ils=InstrumentLineShape(),
            output_wavenumber_cm1=[750.0, 751.0],
        )


def test_padded_result_matches_a_larger_domain_with_edge_spectral_line():
    ils = InstrumentLineShape(fwhm_cm1=0.5)
    step = 0.01
    grid = 740 + np.arange(2201) * step
    signal = 2.0 + 0.02 * (grid - 750) + np.exp(-(((grid - 749.9) / 0.08) ** 2))
    output = np.arange(750.0, 752.01, 0.25)
    margin = ils.margin_cm1(step)
    keep = (grid >= 750 - margin - 1e-9) & (grid <= 752 + margin + 1e-9)
    full = apply_ils_radiance(grid, signal, ils=ils, output_wavenumber_cm1=output)
    cropped = apply_ils_radiance(grid[keep], signal[keep], ils=ils, output_wavenumber_cm1=output)
    np.testing.assert_allclose(cropped, full, atol=2e-14)


def test_kernel_is_centered_normalized_and_symmetric():
    kernel = InstrumentLineShape().kernel(0.01)
    assert kernel.size % 2 == 1
    assert kernel.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(kernel, kernel[::-1])


def test_integer_kernel_support_survives_frequency_subtraction_roundoff():
    step = 0.01
    ils = InstrumentLineShape(
        fwhm_cm1=2 * np.sqrt(2 * np.log(2)) * step * 10 / 6,
        output_step_cm1=step,
        support_sigma=6,
    )
    padding = ils.half_width_samples(step)
    assert padding == 10
    grid = 800 + np.arange(-padding, 101 + padding) * step
    output = 800 + np.arange(101) * step
    values = apply_ils_radiance(grid, np.ones_like(grid), ils=ils, output_wavenumber_cm1=output)
    np.testing.assert_allclose(values, 1.0, atol=2e-15)
