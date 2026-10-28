"""
Tests for continuum.py — the MT_CKD H2O continuum, with known answers.

Offline: the bundled table, a synthetic line and profiles built by hand, so
every expected value follows from the formula.
"""

import math

import numpy as np
import pytest

from pylblrtm.atmosphere import AtmosphericProfile, HomogeneousPath
from pylblrtm.config import load_config, run_forward
from pylblrtm.continuum import h2o_continuum_optical_depth
from pylblrtm.forward import compute_transmittance
from pylblrtm.spectral_data import SpectralData

_C2 = 1.4387752        # second radiation constant [cm K], as in continuum.py
_K_B = 1.380649e-23    # J K⁻¹

# MT_CKD 4.3 at its 1000 cm⁻¹ node, as photonix tabulates it (7 digits)
_NU_NODE, _CS, _CF, _TEXP = 1000.0, 1.331195e-25, 2.450968e-28, 5.635823


def _tau(path, nu):
    return h2o_continuum_optical_depth(np.atleast_1d(nu), path.build_layers())[0]


def _sd():
    """One weak H2O line at 900 cm⁻¹."""
    one = lambda v: np.array([v], dtype=float)
    lines = {"nu0": one(900.0), "Sref": one(1e-21), "g_air": one(0.07),
             "g_self": one(0.35), "Elow": one(200.0), "n_air": one(0.7),
             "shift": one(0.0), "mol_id": np.ones(1, np.int32),
             "iso_id": np.ones(1, np.int32)}
    T = np.arange(100.0, 401.0)
    return SpectralData(lines=lines, tips={(1, 1): (T, 50.0 * (T / 296.0) ** 1.5)},
                        molar_masses={(1, 1): 18.0106}, molecule="H2O")


def _profile(gas="H2O"):
    return AtmosphericProfile(pressure_hpa=[300.0, 700.0, 1013.25],
                              temperature_k=[230.0, 270.0, 295.0],
                              vmr={gas: [1e-4, 5e-3, 2e-2]})


class TestKnownValue:
    """At a table node there is no interpolation: tau is the formula itself."""

    @pytest.mark.parametrize("T_k", [220.0, 296.0, 310.0])
    def test_node_value(self, T_k):
        L, x, p_hpa = 1000.0, 0.01, 900.0
        path = HomogeneousPath(L, {"H2O": x}, pressure_hpa=p_hpa, temperature_k=T_k)
        n_h2o = x * p_hpa * 100.0 / (_K_B * T_k) * L / 1e4
        rho_rat = (p_hpa / 1013.0) * (296.0 / T_k)
        radiation = _NU_NODE * math.tanh(0.5 * _C2 * _NU_NODE / T_k)
        expected = ((_CS * (296.0 / T_k) ** _TEXP * x + _CF * (1.0 - x))
                    * rho_rat * radiation * n_h2o)
        assert _tau(path, _NU_NODE)[0] == pytest.approx(expected, rel=1e-6)


class TestScaling:

    def test_linear_in_path_length(self):
        nu = np.linspace(714.0, 1250.0, 537)
        tau_1 = _tau(HomogeneousPath(100.0, {"H2O": 0.01}), nu)
        tau_2 = _tau(HomogeneousPath(200.0, {"H2O": 0.01}), nu)
        np.testing.assert_allclose(tau_2, 2.0 * tau_1, rtol=1e-14)

    def test_self_and_foreign_split(self):
        """tau / N_H2O = rho·R·[C_s·s·x + C_f·(1 − x)] is affine in x."""
        nu = np.linspace(714.0, 1250.0, 537)
        g = [_tau(HomogeneousPath(100.0, {"H2O": x}), nu) / x for x in (0.01, 0.02, 0.03)]
        np.testing.assert_allclose(g[1], 0.5 * (g[0] + g[2]), rtol=1e-12)

    def test_no_h2o_no_continuum(self):
        tau = _tau(HomogeneousPath(100.0, {"H2O": 0.0}), np.array([900.0, 1000.0]))
        assert np.all(tau == 0.0)


class TestForward:

    def test_compute_transmittance_adds_it(self):
        profile, mu = _profile(), 0.5
        kw = dict(step_cm1=0.01, mu_view=mu)
        nu, t_off = compute_transmittance(_sd(), profile, 880.0, 920.0, **kw)
        _, t_on = compute_transmittance(_sd(), profile, 880.0, 920.0,
                                        h2o_continuum=True, **kw)
        cont = h2o_continuum_optical_depth(nu, profile.build_layers(), _sd()).sum(axis=0)
        np.testing.assert_allclose(t_on, t_off * np.exp(-cont / mu), rtol=1e-12)

    def test_run_forward_matches(self):
        profile = _profile()
        cfg = {"spectral": {"nu_min_cm1": 880.0, "nu_max_cm1": 920.0, "step_cm1": 0.01},
               "output": {"transmittance": False, "optical_depth_layers": True}}
        off = run_forward(_sd(), profile, cfg)
        on = run_forward(_sd(), profile, {**cfg, "calculation": {"h2o_continuum": True}})
        cont = h2o_continuum_optical_depth(off.wavenumber_cm1, profile.build_layers(), _sd())
        np.testing.assert_allclose(on.optical_depth_layers - off.optical_depth_layers,
                                   cont, rtol=1e-12, atol=0.0)

    def test_add_to_is_in_place(self):
        layers = _profile().build_layers()
        nu = np.array([900.0, 1000.0])
        od = np.ones((layers.n_layers, nu.size))
        out = h2o_continuum_optical_depth(nu, layers, add_to=od)
        assert out is od
        np.testing.assert_allclose(od - 1.0, h2o_continuum_optical_depth(nu, layers))


class TestPedestal:
    """Given the lines, the pedestal of the H2O lines comes out of the continuum."""

    def test_value_of_one_line(self):
        """At T_ref and 1 atm: R(ν)·S·N/R(c)·γ/(π(25² + γ²)) within the cutoff, 0 beyond."""
        x = 0.01
        layers = HomogeneousPath(100.0, {"H2O": x}, temperature_k=296.0).build_layers()
        nu = np.array([880.0, 900.0, 924.0, 926.0])
        pure = h2o_continuum_optical_depth(nu, layers)[0]
        net = h2o_continuum_optical_depth(nu, layers, _sd())[0]
        gamma = 0.07 * (1.0 - x) + 0.35 * x
        radiation = lambda v: v * np.tanh(0.5 * _C2 * v / 296.0)
        expected = (radiation(nu) * 1e-21 * layers.column_cm2["H2O"][0] / radiation(900.0)
                    * gamma / (np.pi * (25.0 ** 2 + gamma ** 2)))
        expected[nu > 925.0] = 0.0
        np.testing.assert_allclose(pure - net, expected, rtol=1e-12, atol=0.0)

    def test_lines_have_no_step_at_the_cutoff(self):
        """Lines + net continuum − MT_CKD is AER's line shape V(d) − L(25): it goes to 0 at 25."""
        path = HomogeneousPath(100.0, {"H2O": 0.01})
        sd = _sd()
        sd.lines["nu0"] = np.array([900.0005])            # no grid point exactly at the cutoff
        nu, t_off = compute_transmittance(sd, path, 870.0, 930.0, step_cm1=0.01)
        _, t_on = compute_transmittance(sd, path, 870.0, 930.0, step_cm1=0.01, h2o_continuum=True)
        net_lines = -np.log(t_on) - h2o_continuum_optical_depth(nu, path.build_layers())[0]
        d = np.abs(nu - 900.0005)
        edge = (d > 24.95) & (d < 25.0)
        assert edge.any()
        assert np.all(np.abs(net_lines[edge]) < 5e-3 * -np.log(t_off[edge]))
        np.testing.assert_allclose(net_lines[d > 25.0], 0.0, atol=1e-15)

    def test_only_h2o_lines(self):
        """A CO2 line leaves the H2O continuum as it is."""
        one = lambda v: np.array([v], dtype=float)
        lines = {"nu0": one(900.0), "Sref": one(1e-20), "g_air": one(0.07), "g_self": one(0.09),
                 "Elow": one(200.0), "n_air": one(0.7), "shift": one(0.0),
                 "mol_id": np.full(1, 2, np.int32), "iso_id": np.ones(1, np.int32)}
        T = np.arange(100.0, 401.0)
        sd = SpectralData(lines=lines, tips={(2, 1): (T, 300.0 * T / 296.0)},
                          molar_masses={(2, 1): 43.99}, molecule="CO2")
        layers = HomogeneousPath(100.0, {"H2O": 0.01, "CO2": 4e-4}).build_layers()
        nu = np.array([890.0, 900.0, 910.0])
        np.testing.assert_array_equal(h2o_continuum_optical_depth(nu, layers, sd),
                                      h2o_continuum_optical_depth(nu, layers))

    def test_warns_on_another_cutoff(self):
        with pytest.warns(RuntimeWarning, match="25"):
            compute_transmittance(_sd(), _profile(), 880.0, 920.0, step_cm1=0.01,
                                  line_cutoff_cm1=20.0, h2o_continuum=True)


class TestErrors:

    def test_profile_without_h2o(self):
        with pytest.raises(ValueError, match="H2O"):
            compute_transmittance(_sd(), _profile("CO2"), 880.0, 920.0,
                                  step_cm1=0.01, h2o_continuum=True)

    def test_outside_the_table(self):
        with pytest.raises(ValueError, match="covers"):
            _tau(HomogeneousPath(100.0, {"H2O": 0.01}), np.array([19990.0, 20010.0]))

    def test_config_needs_a_bool(self):
        with pytest.raises(ValueError, match="h2o_continuum"):
            load_config({"calculation": {"h2o_continuum": "yes"}})
