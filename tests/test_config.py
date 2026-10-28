"""
Tests for config.py — per-gas outputs, wavelength bins, attenuation and
run_path() (a standoff scenario described in the YAML).

Offline: synthetic lines, and from_hapi is mocked in run_path.
"""

from unittest.mock import patch

import numpy as np
import pytest

import pylblrtm.data_loader as dl
import pylblrtm.hitran_manager as hm
from pylblrtm.atmosphere import AtmosphericProfile, HomogeneousPath
from pylblrtm.binning import bin_by_wavelength
from pylblrtm.config import load_config, run_forward, run_path
from pylblrtm.spectral_data import SpectralData


def _sd(mol_id, centers, sref=1e-20):
    centers = np.atleast_1d(np.asarray(centers, dtype=float))
    n = centers.size
    lines = {
        "nu0": centers, "Sref": np.full(n, sref), "g_air": np.full(n, 0.07),
        "g_self": np.full(n, 0.07), "Elow": np.full(n, 300.0), "n_air": np.full(n, 0.7),
        "shift": np.zeros(n), "mol_id": np.full(n, mol_id, dtype=np.int32),
        "iso_id": np.ones(n, dtype=np.int32),
    }
    T = np.arange(100.0, 401.0)
    name = {1: "H2O", 2: "CO2", 6: "CH4"}[mol_id]
    return SpectralData(lines=lines, tips={(mol_id, 1): (T, np.full_like(T, 100.0))},
                        molar_masses={(mol_id, 1): 18.0}, molecule=name)


SD = SpectralData.combine([_sd(1, [905.0, 915.0]), _sd(2, [910.0, 925.0])])
PATH = HomogeneousPath(100.0, {"H2O": 1e-3, "CO2": 4e-4})
CFG = {"spectral": {"nu_min_cm1": 900.0, "nu_max_cm1": 930.0, "step_cm1": 0.01}}


def _cfg(**output):
    return {**CFG, "output": {"transmittance": True, **output}}


class TestCombine:

    def test_combine_and_split(self):
        assert SD.mol_ids == (1, 2) and SD.n_lines == 4 and SD.molecule is None
        co2 = SD.for_molecule(2)
        assert co2.n_lines == 2 and list(co2.tips) == [(2, 1)]

    def test_for_missing_molecule_raises(self):
        with pytest.raises(KeyError):
            SD.for_molecule(6)


class TestPerGas:

    def test_product_of_gases_is_total(self):
        res = run_forward(SD, PATH, _cfg(per_gas=True))
        assert set(res.transmittance_per_gas) == {"H2O", "CO2"}
        prod = res.transmittance_per_gas["H2O"] * res.transmittance_per_gas["CO2"]
        np.testing.assert_allclose(prod, res.transmittance, rtol=1e-12)

    def test_total_same_as_without_per_gas(self):
        a = run_forward(SD, PATH, _cfg(per_gas=True)).transmittance
        b = run_forward(SD, PATH, _cfg()).transmittance
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_each_gas_matches_its_own_run(self):
        res = run_forward(SD, PATH, _cfg(per_gas=True))
        alone = run_forward(SD.for_molecule(2), PATH, _cfg()).transmittance
        np.testing.assert_allclose(res.transmittance_per_gas["CO2"], alone, rtol=1e-12)


class TestBinning:

    def test_bins_average_samples(self):
        nu = np.linspace(900.0, 930.0, 3001)
        lam, y = bin_by_wavelength(nu, np.ones_like(nu), 0.02)
        assert np.all(np.diff(lam) > 0) and np.allclose(y, 1.0)
        assert lam[0] == pytest.approx(1e4 / 930.0, abs=0.02)

    def test_bin_is_mean_of_its_samples(self):
        nu = np.linspace(900.0, 930.0, 3001)
        vals = np.sin(nu)
        lam, y = bin_by_wavelength(nu, vals, 0.02)
        lam_s = 1e4 / nu
        i = len(lam) // 2
        sel = (lam_s >= lam[i] - 0.01) & (lam_s < lam[i] + 0.01)
        assert y[i] == pytest.approx(vals[sel].mean(), rel=1e-9)

    def test_several_rows(self):
        nu = np.linspace(900.0, 930.0, 3001)
        _, y = bin_by_wavelength(nu, np.vstack([np.ones_like(nu), 2 * np.ones_like(nu)]), 0.02)
        assert y.shape[0] == 2 and np.allclose(y[1], 2.0)

    def test_range_too_narrow_raises(self):
        with pytest.raises(ValueError, match="no full bin"):
            bin_by_wavelength(np.linspace(1000.0, 1000.5, 51), np.ones(51), 0.02)

    def test_run_forward_bins(self):
        res = run_forward(SD, PATH, _cfg(per_gas=True, bin_um=0.02))
        assert res.wavelength_um.size == res.transmittance_binned.size
        assert set(res.transmittance_per_gas_binned) == {"H2O", "CO2"}
        _, expected = bin_by_wavelength(res.wavenumber_cm1, res.transmittance, 0.02)
        np.testing.assert_allclose(res.transmittance_binned, expected)


class TestAttenuation:

    def test_db_per_metre(self):
        res = run_forward(SD, PATH, _cfg(attenuation_db_m=True))
        expected = -10.0 / 100.0 * np.log10(res.transmittance)
        np.testing.assert_allclose(res.attenuation_db_m, expected, rtol=1e-12)

    def test_per_gas_adds_up_on_the_grid(self):
        res = run_forward(SD, PATH, _cfg(per_gas=True, attenuation_db_m=True))
        total = sum(res.attenuation_per_gas_db_m.values())
        np.testing.assert_allclose(total, res.attenuation_db_m, rtol=1e-9, atol=1e-15)

    def test_needs_a_path_length(self):
        prof = AtmosphericProfile([500.0, 510.0], [280.0, 280.0], {"H2O": [1e-3, 1e-3]})
        with pytest.raises(ValueError, match="HomogeneousPath"):
            run_forward(SD, prof, _cfg(attenuation_db_m=True))


class TestSave:

    def test_csv_writes_grid_and_bins(self, tmp_path):
        res = run_forward(SD, PATH, _cfg(per_gas=True, bin_um=0.02, attenuation_db_m=True))
        res.save(path=str(tmp_path), fmt="csv", prefix="x")
        head = (tmp_path / "x.csv").read_text().splitlines()[0].split(",")
        assert head == ["wavenumber_cm1", "transmittance", "transmittance_H2O", "transmittance_CO2"]
        head_b = (tmp_path / "x_binned.csv").read_text().splitlines()[0].split(",")
        assert head_b[:2] == ["wavelength_um", "transmittance"]
        assert "attenuation_db_m" in head_b and "attenuation_CO2_db_m" in head_b

    def test_npz(self, tmp_path):
        res = run_forward(SD, PATH, _cfg(per_gas=True, bin_um=0.02))
        out = res.save(path=str(tmp_path), fmt="npz", prefix="x")
        data = np.load(out)
        assert "transmittance_CO2" in data and "binned_wavelength_um" in data


class TestPathSection:

    @pytest.mark.parametrize("path, match", [
        ({"gases": {"CO2": 4e-4}}, "length_m"),
        ({"length_m": 10.0}, "gases"),
        ({"length_m": 10.0, "gases": {"CO2": 4e-4}, "gas_units": "ppb"}, "gas_units"),
        ({"length_m": 10.0, "gases": {"CO2": -1}}, "≥ 0"),
        ({"length_m": 10.0, "gases": {"CO2": 4e-4}, "isotopologues": "some"}, "isotopologues"),
        ({"length_m": 10.0, "gases": {"CO2": 4e-4}, "colour": "red"}, "Unknown key"),
    ])
    def test_invalid_path_raises(self, path, match):
        with pytest.raises(ValueError, match=match):
            load_config({**CFG, "path": path})

    @pytest.mark.parametrize("units, value, vmr", [
        ("vmr", 4e-4, 4e-4), ("ppm", 400.0, 4e-4), ("atm", 4e-4 * 0.5, 4e-4),
    ])
    def test_gas_units(self, units, value, vmr):
        from pylblrtm.config import _path_vmr
        pa = {"length_m": 1.0, "pressure_hpa": 1013.25 * 0.5, "gas_units": units,
              "gases": {"CO2": value}}
        assert _path_vmr(pa)["CO2"] == pytest.approx(vmr)


class TestRunPath:

    def _cfg(self, gases, **output):
        return {**CFG, "path": {"length_m": 100.0, "gases": gases},
                "output": {"transmittance": True, **output}}

    def _fake_from_hapi(self, molecule, *a, **kw):
        if molecule == "CO":
            raise hm.HITRANNoLinesError("no lines", molecule="CO")
        return {"H2O": _sd(1, [905.0, 915.0]), "CO2": _sd(2, [910.0, 925.0])}[molecule]

    def test_same_as_run_forward_with_the_path(self):
        with patch.object(dl, "from_hapi", side_effect=self._fake_from_hapi):
            res = run_path(self._cfg({"H2O": 1e-3, "CO2": 4e-4}, per_gas=True))
        ref = run_forward(SD, PATH, _cfg(per_gas=True))
        np.testing.assert_allclose(res.transmittance, ref.transmittance, rtol=1e-12)

    def test_gas_without_lines_is_skipped_with_t_1(self):
        with patch.object(dl, "from_hapi", side_effect=self._fake_from_hapi):
            res = run_path(self._cfg({"H2O": 1e-3, "CO": 1e-7}, per_gas=True, bin_um=0.02,
                                     attenuation_db_m=True))
        assert "CO" in res.skipped_gases
        assert np.all(res.transmittance_per_gas["CO"] == 1.0)
        assert np.all(res.attenuation_per_gas_db_m["CO"] == 0.0)

    def test_no_gas_with_lines_raises(self):
        with patch.object(dl, "from_hapi", side_effect=self._fake_from_hapi):
            with pytest.raises(hm.HITRANNoLinesError):
                run_path(self._cfg({"CO": 1e-7}))

    def test_needs_path_section(self):
        with pytest.raises(ValueError, match="path"):
            run_path(CFG)

    def test_example_yaml_loads(self):
        from pathlib import Path
        cfg = load_config(Path(__file__).resolve().parents[1] / "examples" / "standoff_7gases.yaml")
        assert cfg["path"]["gas_units"] == "atm" and len(cfg["path"]["gases"]) == 7
