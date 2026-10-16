"""Differentiable historical forward used by the frozen z24/Ncol retrieval.

This backend has its own numerical recipe and dense LUT contract. It is not an
adapter for :class:`photonix.lut.CrossSectionLUT` and does not claim equivalence
with the cleaned CPU line-by-line implementation.
"""

from photonix.torch_legacy.continuum import continuum_od_torch, prep_continuum_tables
from photonix.torch_legacy.forward import (
    bt_nu,
    forward_lut,
    integrate_tud,
    integrate_tud_loop,
    interp_sigma,
    planck_nu,
    vmr_to_ncol,
)

RECIPE_ID = "retrieval-z24-ncol-torch-frozen-v1"

__all__ = [
    "RECIPE_ID",
    "bt_nu",
    "continuum_od_torch",
    "forward_lut",
    "integrate_tud",
    "integrate_tud_loop",
    "interp_sigma",
    "planck_nu",
    "prep_continuum_tables",
    "vmr_to_ncol",
]
