"""
Tests for continuum.py — the MT_CKD H2O and CO2 continuum ported from
photonix (Forward_repo/packages/photonix/src/photonix/spectroscopy/continuum.py).
"""

import hashlib
import sys
import types
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

from pylblrtm.atmosphere import HomogeneousPath, _Layers
from pylblrtm.config import run_forward
from pylblrtm.continuum import (ContinuumTables, continuum_coverage,
                                continuum_optical_depth)
from pylblrtm.forward import compute_transmittance
from pylblrtm.spectral_data import SpectralData

_REPO = Path(__file__).resolve().parents[1]
_PHOTONIX = _REPO / "Forward_repo" / "packages" / "photonix" / "src"
_SHA256 = "6ea56d7b59f1533414b9964a05e7bbd2821d856ca476f501f84d5ec2cc956bcd"


def _rad(nu, T):
    """Radiation term with photonix's branches (ν·tanh(u/2); u ≤ 0.01: u·ν/2; u > 10: ν)."""
    u = 1.4387752 * nu / T
    return np.where(u <= 0.01, 0.5 * u * nu, np.where(u > 10.0, nu, nu * np.tanh(0.5 * u)))


def _layers(p=(1.0, 0.8, 0.1), t=(288.2, 260.0, 220.0), xh=(0.0188, 0.012, 5e-6),
            xc=(3.3e-4, 4e-4, 4e-4)):
    air = 2.5e23 * np.arange(1, len(p) + 1)
    xh, xc = np.asarray(xh), np.asarray(xc)
    return _Layers(np.asarray(p, float), np.asarray(t, float), {"H2O": xh, "CO2": xc},
                   {"H2O": air * xh, "CO2": air * xc})


class TestTables:

    def test_bundled_table_is_photonix_table(self):
        payload = files("pylblrtm").joinpath("data/continuum_mtckd.npz").read_bytes()
        assert hashlib.sha256(payload).hexdigest() == _SHA256

    def test_ranges(self):
        tab = ContinuumTables.load()
        assert tab.h2o_range_cm1 == (620.0, 2780.0)
        assert tab.co2_range_cm1[1] == 2000.0
        assert tab.h2o_reference_pressure_hpa == 1013.0


class TestFormula:
    """Recomputed by hand at table nodes, where interpolation is exact."""

    def test_h2o_at_table_nodes(self):
        tab = ContinuumTables.load()
        i = np.arange(20, 200, 30)
        nu = tab.h2o_wavenumber_cm1[i]
        L = _layers()
        got = continuum_optical_depth(nu, L, gases=["H2O"])["H2O"]
        for k in range(L.n_layers):
            T, P, x, N = L.temperature_k[k], L.pressure_atm[k], L.vmr["H2O"][k], L.column_cm2["H2O"][k]
            rad = _rad(nu, T)
            tr = 296.0 / T
            exp = ((tab.h2o_self[i] * tr ** tab.h2o_temperature_exponent[i] * x
                    + tab.h2o_foreign[i] * (1 - x)) * (P * 1013.25 / 1013.0) * tr * rad * N)
            np.testing.assert_allclose(got[k], exp, rtol=1e-13)

    def test_co2_at_table_nodes(self):
        tab = ContinuumTables.load()
        nu = np.array([700.0, 1000.0, 1500.0, 1998.0])
        i = np.searchsorted(tab.co2_wavenumber_cm1, nu)
        assert np.allclose(tab.co2_wavenumber_cm1[i], nu)
        L = _layers()
        got = continuum_optical_depth(nu, L, gases=["CO2"])["CO2"]
        for k in range(L.n_layers):
            T, P, N = L.temperature_k[k], L.pressure_atm[k], L.column_cm2["CO2"][k]
            rad = _rad(nu, T)
            exp = tab.co2_coefficient[i] * N * (P * 1013.25 / 1013.0 * 296.0 / T) * 1e-20 * rad
            np.testing.assert_allclose(got[k], exp, rtol=1e-13)


def _photonix():
    if not _PHOTONIX.is_dir():
        pytest.skip("Forward_repo/photonix not present")
    stub = False
    try:
        import numba  # noqa: F401
    except ImportError:
        # photonix imports numba at package level (its continuum does not use it):
        # a stand-in only while importing, so other tests still see numba missing
        fake = types.ModuleType("numba")
        fake.njit = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))
        fake.prange = range
        sys.modules["numba"] = fake
        stub = True
    sys.path.insert(0, str(_PHOTONIX))
    try:
        from photonix.atmosphere import Layer
        from photonix.spectroscopy.continuum import build_continuum_optical_depth
    finally:
        sys.path.remove(str(_PHOTONIX))
        if stub:
            del sys.modules["numba"]
    return Layer, build_continuum_optical_depth


class TestAgainstPhotonix:

    @pytest.mark.parametrize("gas,flags", [("H2O", {"include_co2": False}),
                                           ("CO2", {"include_h2o": False})])
    def test_bit_identical(self, gas, flags):
        Layer, build = _photonix()
        L = _layers()
        ph = [Layer(k, L.pressure_atm[k], L.temperature_k[k],
                    {g: L.vmr[g][k] for g in L.vmr}, {g: L.column_cm2[g][k] for g in L.vmr})
              for k in range(L.n_layers)]
        nu = np.linspace(714.0, 1250.0, 5361)
        np.testing.assert_array_equal(continuum_optical_depth(nu, L)[gas], build(nu, ph, **flags))


class TestCoverage:

    def test_outside_tables_is_zero(self):
        nu = np.array([600.0, 619.9, 620.0, 1999.9, 2000.0, 2500.0, 2780.0, 2790.0])
        od = continuum_optical_depth(nu, _layers())
        assert np.array_equal(od["H2O"][0] > 0, [False, False, True, True, True, True, True, False])
        assert np.array_equal(od["CO2"][0] > 0, [True, True, True, True, False, False, False, False])
        assert np.array_equal(continuum_coverage(nu, "CO2"), od["CO2"][0] > 0)

    def test_gas_not_in_layers_is_skipped(self):
        L = _layers()
        L.column_cm2.pop("CO2")
        assert set(continuum_optical_depth(np.array([900.0]), L)) == {"H2O"}

    def test_unknown_gas_raises(self):
        with pytest.raises(ValueError, match="No continuum"):
            continuum_optical_depth(np.array([900.0]), _layers(), gases=["O3"])


def _sd():
    n = 2
    T = np.arange(100.0, 401.0)
    lines = {"nu0": np.array([905.0, 915.0]), "Sref": np.full(n, 1e-22),
             "g_air": np.full(n, 0.07), "g_self": np.full(n, 0.3), "Elow": np.full(n, 300.0),
             "n_air": np.full(n, 0.7), "shift": np.zeros(n),
             "mol_id": np.ones(n, dtype=np.int32), "iso_id": np.ones(n, dtype=np.int32)}
    return SpectralData(lines=lines, tips={(1, 1): (T, np.full_like(T, 100.0))},
                        molar_masses={(1, 1): 18.0}, molecule="H2O")


class TestForwardIntegration:

    PATH = HomogeneousPath(100.0, {"H2O": 0.0188}, temperature_k=288.2)

    def test_continuum_adds_to_lines(self):
        nu, t_lines = compute_transmittance(_sd(), self.PATH, 880.0, 940.0, step_cm1=0.01)
        _, t_both = compute_transmittance(_sd(), self.PATH, 880.0, 940.0, step_cm1=0.01,
                                          continuum=True)
        cont = continuum_optical_depth(nu, self.PATH.build_layers())["H2O"][0]
        np.testing.assert_allclose(-np.log(t_both), -np.log(t_lines) + cont, rtol=1e-10)
        assert cont.min() > 0

    def test_per_gas_includes_continuum(self):
        cfg = {"spectral": {"nu_min_cm1": 880.0, "nu_max_cm1": 940.0, "step_cm1": 0.01},
               "calculation": {"continuum": True},
               "output": {"transmittance": True, "per_gas": True}}
        res = run_forward(_sd(), self.PATH, cfg)
        np.testing.assert_allclose(res.transmittance_per_gas["H2O"], res.transmittance, rtol=1e-12)
        _, t_both = compute_transmittance(_sd(), self.PATH, 880.0, 940.0, step_cm1=0.01,
                                          continuum=True)
        np.testing.assert_allclose(res.transmittance, t_both, rtol=1e-12)

    def test_continuum_doubles_with_length(self):
        L1 = HomogeneousPath(100.0, {"H2O": 0.0188}).build_layers()
        L2 = HomogeneousPath(200.0, {"H2O": 0.0188}).build_layers()
        nu = np.linspace(800.0, 1200.0, 41)
        np.testing.assert_allclose(continuum_optical_depth(nu, L2)["H2O"],
                                   2 * continuum_optical_depth(nu, L1)["H2O"], rtol=1e-12)

    def test_yaml_continuum_must_be_bool(self):
        from pylblrtm.config import load_config
        with pytest.raises(ValueError, match="continuum"):
            load_config({"calculation": {"continuum": "yes"}})

    def test_default_is_off(self):
        from pylblrtm.config import load_config
        assert load_config({})["calculation"]["continuum"] is False
