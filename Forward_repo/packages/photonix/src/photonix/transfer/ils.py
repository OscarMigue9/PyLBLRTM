"""Gaussian instrumental convolution on a padded radiance grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from photonix._validation import finite_vector, positive_scalar, spectral_grid


@dataclass(frozen=True)
class InstrumentLineShape:
    """Gaussian ILS: FWHM and output step [cm^-1], finite support in sigma.

    The forward calculates this margin before integrating. Instrumental
    convolution is applied only after combining atmospheric and surface terms.
    """

    fwhm_cm1: float = 0.5
    output_step_cm1: float = 0.25
    support_sigma: float = 6.0

    def __post_init__(self) -> None:
        for name in ("fwhm_cm1", "output_step_cm1", "support_sigma"):
            object.__setattr__(self, name, positive_scalar(getattr(self, name), name))

    def half_width_samples(self, step_cm1: float) -> int:
        """Return sampled half-support; this same value defines the padding."""
        step = positive_scalar(step_cm1, "step_cm1")
        sigma = self.fwhm_cm1 / (2 * np.sqrt(2 * np.log(2)))
        sample_count = self.support_sigma * sigma / step
        # Subtracting large neighboring wavenumbers slightly perturbs the step.
        # Preserve an integer support instead of spuriously adding a sample.
        nearest = round(sample_count)
        if abs(sample_count - nearest) <= 1e-9 * max(1.0, sample_count):
            sample_count = nearest
        return max(1, int(np.ceil(sample_count)))

    def margin_cm1(self, step_cm1: float) -> float:
        return self.half_width_samples(step_cm1) * step_cm1

    def kernel(self, step_cm1: float) -> np.ndarray:
        """Return a centered, normalized sampled Gaussian."""
        half_width = self.half_width_samples(step_cm1)
        coordinates = np.arange(-half_width, half_width + 1) * step_cm1
        sigma = self.fwhm_cm1 / (2 * np.sqrt(2 * np.log(2)))
        kernel = np.exp(-0.5 * (coordinates / sigma) ** 2)
        return kernel / kernel.sum()


def apply_ils_radiance(
    wavenumber_cm1, radiance, *, ils: InstrumentLineShape, output_wavenumber_cm1
) -> np.ndarray:
    """Convolve radiance, then sample inside the fully supported domain.

    Missing padding is an error. This never fabricates out-of-band radiance by
    zero padding, endpoint extrapolation or renormalizing a truncated kernel.
    """
    wavenumber = spectral_grid(wavenumber_cm1, uniform=True)
    output_grid = spectral_grid(output_wavenumber_cm1)
    values = finite_vector(radiance, "radiance")
    if values.shape != wavenumber.shape:
        raise ValueError("radiance must have one value per wavenumber_cm1")
    if not isinstance(ils, InstrumentLineShape):
        raise TypeError("ils must be an InstrumentLineShape")
    step = float(wavenumber[1] - wavenumber[0])
    margin = ils.margin_cm1(step)
    tolerance = max(1e-8, step * 1e-6)
    if (
        output_grid[0] < wavenumber[0] + margin - tolerance
        or output_grid[-1] > wavenumber[-1] - margin + tolerance
    ):
        raise ValueError(
            f"radiance grid needs at least {margin:g} cm^-1 of ILS padding on each side"
        )
    convolved = fftconvolve(values, ils.kernel(step), mode="same")
    return np.interp(output_grid, wavenumber, convolved)
