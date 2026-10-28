"""
binning.py — Spectral sampling of monochromatic results onto wavelength bins.

The forward model works on a uniform wavenumber grid; a sensor or a comparison
with another model often needs the spectrum averaged over bins of constant
width in wavelength (e.g. 0.02 µm).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def bin_by_wavelength(
    wavenumber_cm1: np.ndarray,
    values: np.ndarray,
    delta_um: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average *values*(ν) over wavelength bins of width *delta_um*.

    Bin edges are multiples of *delta_um* inside the covered wavelength range
    (from ⌈λ_min/Δ⌉·Δ to ⌊λ_max/Δ⌋·Δ). Each bin is the mean of the samples with
    edge_left ≤ λ < edge_right; a bin with no sample takes the value
    interpolated at its centre.

    Parameters
    ----------
    wavenumber_cm1 : array (n,)
        Wavenumber grid [cm⁻¹], any order.
    values : array (n,) or (m, n)
        Quantity on that grid (e.g. transmittance); several rows are binned
        independently.
    delta_um : float
        Bin width [µm].

    Returns
    -------
    lambda_centers_um : array (n_bins,)
        Bin centres [µm], increasing.
    binned : array (n_bins,) or (m, n_bins)
        Mean of *values* in each bin.
    """
    if not (np.isfinite(delta_um) and delta_um > 0):
        raise ValueError(f"delta_um must be positive; got {delta_um!r}")
    nu = np.asarray(wavenumber_cm1, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    one_row = vals.ndim == 1
    vals = np.atleast_2d(vals)
    if vals.shape[1] != nu.size:
        raise ValueError("values must have the same length as wavenumber_cm1")

    lam = 1e4 / nu
    order = np.argsort(lam)
    lam = lam[order]
    vals = vals[:, order]

    lam_min = math.ceil(lam[0] / delta_um) * delta_um
    lam_max = math.floor(lam[-1] / delta_um) * delta_um
    edges = np.arange(lam_min, lam_max + 1e-9, delta_um)
    if edges.size < 2:
        raise ValueError(
            f"the range {lam[0]:.4f}–{lam[-1]:.4f} µm holds no full bin of {delta_um} µm"
        )
    centers = 0.5 * (edges[:-1] + edges[1:])
    left = np.searchsorted(lam, edges[:-1], side="left")
    right = np.searchsorted(lam, edges[1:], side="left")

    binned = np.empty((vals.shape[0], centers.size))
    for i, (lo, hi) in enumerate(zip(left, right)):
        if hi > lo:
            binned[:, i] = vals[:, lo:hi].mean(axis=1)
        else:
            binned[:, i] = [np.interp(centers[i], lam, row) for row in vals]
    return centers, (binned[0] if one_row else binned)
