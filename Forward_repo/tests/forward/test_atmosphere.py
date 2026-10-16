import numpy as np
import pytest

from photonix.atmosphere import AtmosphericProfile, Layer
from photonix.constants import AVOGADRO_MOL_INV, DRY_AIR_MOLAR_MASS_KG_MOL, GRAVITY_M_S2
from photonix.spectroscopy.species import default_species


def test_hydrostatic_column_and_layer_order():
    profile = AtmosphericProfile([10, 100, 1000], [210, 250, 290], {"CO2": [4e-4] * 3})
    layers = profile.build_layers(default_species(["CO2"]))
    assert len(layers) == 2
    np.testing.assert_allclose([layer.temperature_k for layer in layers], [230, 270])
    expected_column = (
        AVOGADRO_MOL_INV / (GRAVITY_M_S2 * DRY_AIR_MOLAR_MASS_KG_MOL) * 990 * 100 / 1e4 * 4e-4
    )
    assert sum(layer.column_density_cm2["CO2"] for layer in layers) == pytest.approx(
        expected_column
    )
    assert layers[0].pressure_atm < layers[1].pressure_atm


@pytest.mark.parametrize(
    "pressure,temperature,vmr",
    [
        ([1000, 100], [280, 220], {"H2O": [0.01, 0.001]}),
        ([0, 100], [220, 280], {"H2O": [0, 0.01]}),
        ([10, 100], [220, np.nan], {"H2O": [0, 0.01]}),
        ([10, 100], [220, 280], {"H2O": [0.01]}),
        ([10, 100], [220, 280], {"H2O": [-0.01, 0.01]}),
        ([10, 100], [220, 280], {"H2O": [0.9, 0.9], "CO2": [0.2, 0.2]}),
    ],
)
def test_invalid_profile_is_rejected(pressure, temperature, vmr):
    with pytest.raises(ValueError):
        AtmosphericProfile(pressure, temperature, vmr)


def test_missing_gas_cannot_become_transparent():
    profile = AtmosphericProfile([10, 100], [230, 280], {"H2O": [0, 0.01]})
    with pytest.raises(ValueError, match="missing vmr"):
        profile.build_layers(default_species(["CH4"]))


def test_input_mutation_does_not_change_profile():
    pressure = np.array([10.0, 100.0])
    water = np.array([0.0, 0.01])
    profile = AtmosphericProfile(pressure, [230, 280], {"H2O": water})
    pressure[0] = -1
    water[:] = 1
    assert profile.pressure_hpa[0] == 10
    assert profile.vmr["H2O"][0] == 0
    with pytest.raises(ValueError):
        profile.temperature_k[0] = 0


def test_invalid_layer_never_reaches_spectroscopy():
    with pytest.raises(ValueError):
        Layer(0, 1.0, 280, {"H2O": 0.01}, {"H2O": np.inf})
