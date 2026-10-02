"""
Tests for data_loader.py — from_hapi and from_files.

All network and HAPI calls are mocked.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pylblrtm.hitran_manager as hm
import pylblrtm.data_loader as dl
from pylblrtm.spectral_data import SpectralData


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_lines(*iso_ids: int, mol_id: int = 1, n: int = 5) -> dict:
    """Lines dict with n entries per isotopologue, concatenated."""
    chunks = []
    for iso in iso_ids:
        chunk = {
            "nu0":    np.linspace(700.0, 1400.0, n),
            "Sref":   np.ones(n) * 1e-23,
            "g_air":  np.ones(n) * 0.07,
            "g_self": np.ones(n) * 0.35,
            "Elow":   np.ones(n) * 100.0,
            "n_air":  np.ones(n) * 0.67,
            "shift":  np.zeros(n),
            "mol_id": np.full(n, mol_id, dtype=np.int32),
            "iso_id": np.full(n, iso, dtype=np.int32),
        }
        chunks.append(chunk)
    if len(chunks) == 1:
        return chunks[0]
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}


def _dummy_tips(T_min=150.0, T_max=400.0):
    T = np.arange(T_min, T_max + 1.0)
    return T, np.ones_like(T) * 174.58


# ── from_hapi ─────────────────────────────────────────────────────────────────

class TestFromHapi:

    def _patch_hapi(self, lines, tips_return=None, molar_mass=18.01):
        tips_return = tips_return or _dummy_tips()
        return [
            patch.object(hm, "get_line_parameters", return_value=lines),
            patch.object(hm, "get_tips_table", return_value=tips_return),
            patch.object(hm, "get_molar_mass", return_value=molar_mass),
        ]

    def test_single_iso_returns_spectral_data(self):
        lines = _make_lines(1)
        patches = self._patch_hapi(lines)
        with patches[0], patches[1], patches[2]:
            sd = dl.from_hapi("H2O", 700, 1400, iso_id=1)
        assert isinstance(sd, SpectralData)
        assert (1, 1) in sd.tips
        assert (1, 1) in sd.molar_masses

    def test_molecule_label_set_correctly(self):
        lines = _make_lines(1)
        patches = self._patch_hapi(lines)
        with patches[0], patches[1], patches[2]:
            sd = dl.from_hapi("H2O", 700, 1400)
        assert sd.molecule == "H2O"

    def test_multi_iso_builds_tips_per_isotopologue(self):
        lines = _make_lines(1, 2, 3)
        tips_data = _dummy_tips()
        with patch.object(hm, "get_line_parameters", return_value=lines), \
             patch.object(hm, "get_tips_table", return_value=tips_data) as mock_tips, \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_hapi("H2O", 700, 1400, iso_id=[1, 2, 3])

        assert mock_tips.call_count == 3
        assert (1, 1) in sd.tips
        assert (1, 2) in sd.tips
        assert (1, 3) in sd.tips

    def test_n_lines_matches_downloaded_data(self):
        lines = _make_lines(1, 2, n=7)  # 14 lines total
        patches = self._patch_hapi(lines)
        with patches[0], patches[1], patches[2]:
            sd = dl.from_hapi("H2O", 700, 1400, iso_id=[1, 2])
        assert sd.n_lines == 14

    def test_molar_mass_stored_per_iso(self):
        lines = _make_lines(1, 2)
        masses = {1: 18.010565, 2: 20.014811}
        call_count = [0]

        def mock_mass(mol, iso, *a, **kw):
            call_count[0] += 1
            return masses[iso]

        with patch.object(hm, "get_line_parameters", return_value=lines), \
             patch.object(hm, "get_tips_table", return_value=_dummy_tips()), \
             patch.object(hm, "get_molar_mass", side_effect=mock_mass):
            sd = dl.from_hapi("H2O", 700, 1400, iso_id=[1, 2])

        assert sd.molar_masses[(1, 1)] == pytest.approx(18.010565)
        assert sd.molar_masses[(1, 2)] == pytest.approx(20.014811)

    def test_passes_hapi_kwargs_to_get_line_parameters(self):
        lines = _make_lines(1)
        with patch.object(hm, "get_line_parameters", return_value=lines) as mock_glp, \
             patch.object(hm, "get_tips_table", return_value=_dummy_tips()), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            dl.from_hapi("H2O", 700, 1400, force_download=True)

        assert mock_glp.call_args[1]["force_download"] is True

    def test_all_iso_id_accepted(self):
        """iso_id='all' is passed through to get_line_parameters."""
        lines = _make_lines(1, 2)
        with patch.object(hm, "get_line_parameters", return_value=lines) as mock_glp, \
             patch.object(hm, "get_tips_table", return_value=_dummy_tips()), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            dl.from_hapi("H2O", 700, 1400, iso_id="all")

        assert mock_glp.call_args[1]["iso_id"] == "all"


# ── from_files ────────────────────────────────────────────────────────────────

class TestFromFiles:

    def test_single_iso_from_files(self, tmp_path):
        T, Q = _dummy_tips()
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_files(
                "dummy.par",
                tips={(1, 1): (T, Q)},
            )
        assert isinstance(sd, SpectralData)
        assert (1, 1) in sd.tips
        assert sd.n_lines == 5

    def test_missing_tips_raises_value_error(self):
        lines = _make_lines(1, 2)  # has iso 1 and 2
        with patch.object(hm, "par_to_dict", return_value=lines):
            with pytest.raises(ValueError, match="Missing TIPS"):
                dl.from_files(
                    "dummy.par",
                    tips={(1, 1): _dummy_tips()},  # iso 2 missing
                )

    def test_tips_from_path(self, tmp_path):
        tips_file = tmp_path / "tips.txt"
        T, Q = _dummy_tips()
        tips_file.write_text(
            "\n".join(f"{t:.1f}  {q:.6f}" for t, q in zip(T, Q)),
            encoding="utf-8",
        )
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_files("dummy.par", tips={(1, 1): tips_file})
        assert (1, 1) in sd.tips
        np.testing.assert_allclose(sd.tips[(1, 1)][0], T)

    def test_tips_from_arrays(self):
        T, Q = _dummy_tips()
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_files("dummy.par", tips={(1, 1): (T, Q)})
        np.testing.assert_allclose(sd.tips[(1, 1)][1], Q)

    def test_manual_molar_masses_used(self):
        T, Q = _dummy_tips()
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines):
            sd = dl.from_files(
                "dummy.par",
                tips={(1, 1): (T, Q)},
                molar_masses={(1, 1): 18.999},
            )
        assert sd.molar_masses[(1, 1)] == pytest.approx(18.999)

    def test_molar_mass_auto_from_hapi(self):
        T, Q = _dummy_tips()
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.010565) as mock_mm:
            sd = dl.from_files("dummy.par", tips={(1, 1): (T, Q)})
        mock_mm.assert_called_once_with("H2O", 1)
        assert sd.molar_masses[(1, 1)] == pytest.approx(18.010565)

    def test_multi_iso_from_files(self):
        T, Q = _dummy_tips()
        lines = _make_lines(1, 2)
        tips = {(1, 1): (T, Q), (1, 2): (T, Q)}
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", side_effect=[18.01, 20.01]):
            sd = dl.from_files("dummy.par", tips=tips)
        assert (1, 1) in sd.tips
        assert (1, 2) in sd.tips
        assert sd.n_lines == 10

    def test_extra_tips_keys_silently_ignored(self):
        """Tips entries for iso_ids not in the .par are ignored."""
        T, Q = _dummy_tips()
        lines = _make_lines(1)  # only iso 1
        tips = {(1, 1): (T, Q), (1, 99): (T, Q)}  # iso 99 not in .par
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_files("dummy.par", tips=tips)
        assert (1, 99) not in sd.tips

    def test_molecule_label_single_molecule(self):
        T, Q = _dummy_tips()
        lines = _make_lines(1)
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", return_value=18.01):
            sd = dl.from_files("dummy.par", tips={(1, 1): (T, Q)})
        assert sd.molecule == "H2O"

    def test_molecule_label_none_for_multi_molecule(self):
        """Two different molecules → molecule=None."""
        T, Q = _dummy_tips()
        # mol_id=1 (H2O) and mol_id=2 (CO2)
        lines_h2o = _make_lines(1, mol_id=1)
        lines_co2 = _make_lines(1, mol_id=2)
        lines = {k: np.concatenate([lines_h2o[k], lines_co2[k]]) for k in lines_h2o}
        tips = {(1, 1): (T, Q), (2, 1): (T, Q)}
        with patch.object(hm, "par_to_dict", return_value=lines), \
             patch.object(hm, "get_molar_mass", side_effect=[18.01, 44.01]):
            sd = dl.from_files("dummy.par", tips=tips)
        assert sd.molecule is None
