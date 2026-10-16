"""Small frozen fixtures for the Torch recipe used by the final checkpoint."""

from __future__ import annotations

import hashlib
from importlib.resources import files

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from photonix.torch_legacy import (  # noqa: E402
    RECIPE_ID,
    continuum_od_torch,
    forward_lut,
    integrate_tud,
    integrate_tud_loop,
    interp_sigma,
    planck_nu,
    prep_continuum_tables,
)


def _fixture():
    dtype = torch.float64
    nu = torch.tensor([714.0, 900.0, 1250.0], dtype=dtype)
    temperature_axis = torch.tensor([200.0, 300.0], dtype=dtype)
    log_pressure_axis = torch.tensor([-4.0, 0.0], dtype=dtype)
    log_sigma = torch.tensor(
        [
            [
                [[-24.0, -23.5, -24.3], [-23.8, -23.3, -24.1]],
                [[-23.9, -23.4, -24.2], [-23.7, -23.2, -24.0]],
            ]
        ],
        dtype=dtype,
    )
    layer_temperatures = torch.tensor([[250.0, 270.0]], dtype=dtype)
    layer_pressures = torch.tensor([[0.1, 0.7]], dtype=dtype)
    gas_columns = torch.tensor([[[2e22, 4e22]]], dtype=dtype)
    level_temperatures = torch.tensor([[230.0, 260.0, 290.0]], dtype=dtype)
    skin_temperature = torch.tensor([295.0], dtype=dtype)
    emissivity = torch.tensor([[0.92, 0.85, 0.97]], dtype=dtype)
    mu = torch.tensor([0.8], dtype=dtype)
    return (
        nu,
        temperature_axis,
        log_pressure_axis,
        log_sigma,
        layer_temperatures,
        layer_pressures,
        gas_columns,
        level_temperatures,
        skin_temperature,
        emissivity,
        mu,
    )


def test_forward_matches_frozen_historical_output():
    """Reference values were computed with model commit 9f3e0a4's Torch file."""
    args = _fixture()
    nu, temperature_axis, log_pressure_axis, log_sigma = args[:4]
    layer_temperatures, layer_pressures = args[4:6]
    actual_sigma = interp_sigma(
        log_sigma[0],
        temperature_axis,
        log_pressure_axis,
        layer_temperatures,
        torch.log10(layer_pressures),
    )
    np.testing.assert_allclose(
        actual_sigma.numpy(),
        [
            [
                [1.5848931924611108e-24, 5.011872336272715e-24, 7.943282347242789e-25],
                [1.8291735044305426e-24, 5.784354509632611e-24, 9.167584085098452e-25],
            ]
        ],
        rtol=2e-13,
    )
    result = forward_lut(
        log_sigma,
        temperature_axis,
        log_pressure_axis,
        layer_temperatures,
        layer_pressures,
        args[6],
        0.0,
        args[7],
        args[8],
        args[9],
        nu,
        mu=args[10],
    )
    expected = {
        "T": [0.8771467182142255, 0.6606609313595734, 0.9364153730722069],
        "U": [11.596113759814788, 22.974037901836244, 1.8579849081759732],
        "D": [15.383006692073783, 30.918407605363445, 2.480665914565333],
        "L_obs": [123.61455927179158, 87.29334069402739, 49.59213536283525],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(result[name].numpy()[0], values, rtol=2e-13)
    assert RECIPE_ID == "retrieval-z24-ncol-torch-frozen-v1"


def test_vectorized_tud_matches_layer_loop_and_preserves_downward_angle():
    nu, *rest = _fixture()
    level_temperatures = rest[6]
    planck_levels = planck_nu(nu, level_temperatures)
    depth = torch.tensor(
        [[[1e-10, 0.07, 0.8], [0.004, 0.3, 2.5]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    angled = integrate_tud(depth, planck_levels, torch.tensor([0.6]).view(1, 1, 1))
    loop = integrate_tud_loop(depth, planck_levels, torch.tensor([0.6]).view(1, 1, 1))
    for vectorized, reference in zip(angled, loop):
        torch.testing.assert_close(vectorized, reference, rtol=1e-14, atol=1e-14)
    nadir = integrate_tud(depth, planck_levels)
    torch.testing.assert_close(angled[1], nadir[1], rtol=0, atol=0)
    angled[0].sum().backward()
    assert torch.isfinite(depth.grad).all()


def test_historical_continuum_table_and_values():
    resource = files("photonix.torch_legacy").joinpath("continuum_tables.npz")
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == (
        "192e5e2771947e371d729df5d0f08af8bed8804f52f3f7c818a12851bdfcef29"
    )
    nu = np.array([714.0, 900.0, 1250.0])
    self_coefficient, foreign_coefficient, exponent = prep_continuum_tables(nu)
    np.testing.assert_allclose(
        self_coefficient,
        [8.915971629278674e-25, 2.5974899999999968e-25, 9.220266999999973e-26],
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        foreign_coefficient,
        [2.8526178872858743e-27, 5.480877000000009e-28, 1.1699960000000008e-27],
        rtol=2e-13,
    )
    np.testing.assert_allclose(exponent, [3.8151286, 5.276027, 5.90055], rtol=2e-13)
    args = _fixture()
    result = continuum_od_torch(
        args[0],
        torch.from_numpy(self_coefficient),
        torch.from_numpy(foreign_coefficient),
        torch.from_numpy(exponent),
        args[4],
        args[5],
        torch.tensor([[0.01, 0.02]], dtype=torch.float64),
        torch.tensor([[1e20, 3e20]], dtype=torch.float64),
    )
    np.testing.assert_allclose(
        result.numpy()[0],
        [
            [0.00016203427549082351, 7.243844402687164e-05, 5.402910432748238e-05],
            [0.004420896790790299, 0.0018291873533803538, 0.0012396970901922635],
        ],
        rtol=2e-13,
    )
