"""
Tests for hitran_manager.py — multi-isotopologue support.

All tests are offline: HAPI and network calls are mocked.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

import pylblrtm.hitran_manager as hm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_lines(n: int, iso_id: int, mol_id: int = 1) -> dict:
    """Dummy line parameter dict with n entries for a given isotopologue."""
    return {
        "nu0":    np.linspace(700.0, 1400.0, n),
        "Sref":   np.ones(n) * 1e-23,
        "g_air":  np.ones(n) * 0.07,
        "g_self": np.ones(n) * 0.35,
        "Elow":   np.ones(n) * 100.0,
        "n_air":  np.ones(n) * 0.67,
        "shift":  np.zeros(n),
        "mol_id": np.full(n, mol_id, dtype=np.int32),
        "iso_id": np.full(n, iso_id, dtype=np.int32),
    }


@pytest.fixture
def mock_hapi():
    """Minimal HAPI mock. H2O (mol_id=1) has 4 isotopologues."""
    m = MagicMock()
    m.ISO = {(1, 1): None, (1, 2): None, (1, 3): None, (1, 4): None}
    m.HAPI_VERSION = "1.3.0"
    return m


# ── _fetch_single_iso ─────────────────────────────────────────────────────────

class TestFetchSingleIso:

    def test_cache_hit_skips_download(self, tmp_path, mock_hapi):
        """Cached data is returned without calling fetch()."""
        expected = _make_lines(10, 1)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_extract_line_data", return_value=expected):

            result = hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

            mock_hapi.fetch.assert_not_called()
            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    def test_force_download_ignores_cache(self, tmp_path, mock_hapi):
        """force_download=True calls fetch() even when the table is cached."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_write_metadata"), \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(5, 1)):

            hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, True)

            mock_hapi.fetch.assert_called_once()

    def test_no_internet_no_cache_raises(self, tmp_path, mock_hapi):
        """No internet and no cache → HITRANConnectionError."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=False):

            with pytest.raises(hm.HITRANConnectionError):
                hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

    def test_no_internet_force_download_falls_back_to_cache(self, tmp_path, mock_hapi):
        """force_download=True + no internet + existing cache → uses cache."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=True), \
             patch.object(hm, "_can_reach_hitran", return_value=False), \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(5, 1)):

            result = hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, True)

            mock_hapi.fetch.assert_not_called()
            assert len(result["nu0"]) == 5

    def test_download_writes_metadata(self, tmp_path, mock_hapi):
        """A successful download also calls _write_metadata."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_write_metadata") as mock_meta, \
             patch.object(hm, "_extract_line_data", return_value=_make_lines(3, 1)):

            hm._fetch_single_iso("H2O", 1, 1, 700.0, 1400.0, tmp_path, False)

            mock_meta.assert_called_once()


# ── get_line_parameters ───────────────────────────────────────────────────────

class TestGetLineParameters:

    def test_unknown_molecule_raises_value_error(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(ValueError, match="Unknown molecule"):
                hm.get_line_parameters("FAKE", 700, 1400, cache_dir=tmp_path)

    def test_invalid_iso_id_string_raises(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(ValueError, match="'all'"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id="none", cache_dir=tmp_path)

    def test_empty_iso_list_raises(self, tmp_path, mock_hapi):
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso"):
            with pytest.raises(hm.HITRANDataError, match="No isotopologues"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id=[], cache_dir=tmp_path)

    # ── iso_id = int ──────────────────────────────────────────────────────────

    def test_int_iso_id_calls_fetch_once(self, tmp_path, mock_hapi):
        """iso_id=2 → _fetch_single_iso called once with iso_id=2."""
        expected = _make_lines(10, 2)
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=expected) as mock_fetch:

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=2, cache_dir=tmp_path)

            mock_fetch.assert_called_once()
            assert mock_fetch.call_args[0][2] == 2  # iso_id argument
            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    def test_default_iso_id_is_1(self, tmp_path, mock_hapi):
        """Default iso_id=1 when not specified."""
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=_make_lines(5, 1)) as mock_fetch:

            hm.get_line_parameters("H2O", 700, 1400, cache_dir=tmp_path)

            assert mock_fetch.call_args[0][2] == 1

    # ── iso_id = list ─────────────────────────────────────────────────────────

    def test_list_iso_id_concatenates_arrays(self, tmp_path, mock_hapi):
        """iso_id=[1, 2] → arrays concatenated, total length = sum of both."""
        lines_1 = _make_lines(10, 1)
        lines_2 = _make_lines(8, 2)

        def _side(mol, mol_id, iso_id, *a, **kw):
            return lines_1 if iso_id == 1 else lines_2

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[1, 2], cache_dir=tmp_path)

            assert len(result["nu0"]) == 18
            assert set(result["iso_id"].tolist()) == {1, 2}

    def test_all_result_keys_have_same_length(self, tmp_path, mock_hapi):
        """All arrays in the result dict must be the same length."""
        lines_1 = _make_lines(7, 1)
        lines_2 = _make_lines(13, 2)

        def _side(mol, mol_id, iso_id, *a, **kw):
            return lines_1 if iso_id == 1 else lines_2

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[1, 2], cache_dir=tmp_path)

            lengths = {k: len(v) for k, v in result.items()}
            assert len(set(lengths.values())) == 1, f"Inconsistent lengths: {lengths}"
            assert list(lengths.values())[0] == 20

    def test_single_item_list_returns_without_concatenation(self, tmp_path, mock_hapi):
        """iso_id=[3] (single-element list) returns the same dict as iso_id=3."""
        expected = _make_lines(5, 3)

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", return_value=expected):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id=[3], cache_dir=tmp_path)

            np.testing.assert_array_equal(result["iso_id"], expected["iso_id"])

    # ── iso_id = "all" ────────────────────────────────────────────────────────

    def test_all_discovers_isotopologues_from_hapi_iso(self, tmp_path, mock_hapi):
        """iso_id='all' uses mock_hapi.ISO to discover iso_ids [1,2,3,4]."""
        per_iso = {i: _make_lines(5, i) for i in [1, 2, 3, 4]}

        def _side(mol, mol_id, iso_id, *a, **kw):
            return per_iso[iso_id]

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)

            assert len(result["nu0"]) == 20
            assert sorted(np.unique(result["iso_id"]).tolist()) == [1, 2, 3, 4]

    def test_all_case_insensitive(self, tmp_path, mock_hapi):
        """iso_id='ALL' and iso_id='All' are accepted."""
        per_iso = {i: _make_lines(3, i) for i in [1, 2, 3, 4]}

        def _side(mol, mol_id, iso_id, *a, **kw):
            return per_iso[iso_id]

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            result = hm.get_line_parameters("H2O", 700, 1400, iso_id="ALL", cache_dir=tmp_path)
            assert len(result["nu0"]) == 12

    def test_all_processes_in_ascending_order(self, tmp_path, mock_hapi):
        """iso_id='all' processes isotopologues sorted in ascending order."""
        call_order = []

        def _side(mol, mol_id, iso_id, *a, **kw):
            call_order.append(iso_id)
            return _make_lines(2, iso_id)

        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "_fetch_single_iso", side_effect=_side):

            hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)

            assert call_order == sorted(call_order)

    def test_all_no_isotopologues_raises(self, tmp_path, mock_hapi):
        """If HAPI.ISO has no entries for the molecule, raise HITRANDataError."""
        mock_hapi.ISO = {}  # vacío — ningún isotopólogo conocido

        with patch.object(hm, "_get_hapi", return_value=mock_hapi):
            with pytest.raises(hm.HITRANDataError, match="No isotopologues"):
                hm.get_line_parameters("H2O", 700, 1400, iso_id="all", cache_dir=tmp_path)


# ── "Failed to retrieve data" → error específico ──────────────────────────────

class TestNoDataErrors:
    """HITRAN answers 404 both for an empty range and for an unserved isotopologue."""

    def _fetch_failing(self, tmp_path, mock_hapi, serves):
        mock_hapi.ISO = {(1, 1): [1, "H2(16O)", 0.997, 18.01, "H2O"]}
        mock_hapi.fetch.side_effect = Exception("Failed to retrieve data for given parameters.")
        with patch.object(hm, "_get_hapi", return_value=mock_hapi), \
             patch.object(hm, "is_cached", return_value=False), \
             patch.object(hm, "_can_reach_hitran", return_value=True), \
             patch.object(hm, "_hitran_serves", return_value=serves):
            hm._fetch_single_iso("H2O", 1, 1, 714.0, 1250.0, tmp_path, False)

    def test_served_elsewhere_is_no_lines(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNoLinesError, match="no lines"):
            self._fetch_failing(tmp_path, mock_hapi, serves=True)

    def test_not_served_at_all(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANNotAvailableError, match="does not serve"):
            self._fetch_failing(tmp_path, mock_hapi, serves=False)

    def test_unknown_stays_generic_data_error(self, tmp_path, mock_hapi):
        with pytest.raises(hm.HITRANDataError) as info:
            self._fetch_failing(tmp_path, mock_hapi, serves=None)
        assert not isinstance(info.value, (hm.HITRANNoLinesError, hm.HITRANNotAvailableError))

    def test_new_errors_are_data_errors(self):
        """Code that catches HITRANDataError keeps working."""
        assert issubclass(hm.HITRANNoLinesError, hm.HITRANDataError)
        assert issubclass(hm.HITRANNotAvailableError, hm.HITRANDataError)


# ── TIPS: rango de la tabla y versión de respaldo ─────────────────────────────

@pytest.fixture
def tips_hapi():
    """HAPI mock: (1,1) has TIPS-2021 up to 1000 K; (1,2) only has TIPS-2025."""
    m = MagicMock()
    m.TIPS_2021_ISOT_HASH = {(1, 1): np.arange(1.0, 1001.0)}
    m.TIPS_2025_ISOT_HASH = {(1, 1): np.arange(1.0, 1001.0), (1, 2): np.arange(1.0, 3001.0)}
    m.TIPS_2017_ISOT_HASH = {}
    m.partitionSum.side_effect = lambda M, I, T, version=2021: float(T)
    return m


class TestTipsTable:

    def test_clipped_to_table_max(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, Q = hm.get_tips_table("H2O", 1, T_min=150, T_max=3000)
        assert T[0] == 150.0 and T[-1] == 1000.0
        assert np.array_equal(T, Q)

    def test_inside_table_unchanged(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 1, T_min=150, T_max=400)
        assert T[0] == 150.0 and T[-1] == 400.0 and len(T) == 251

    def test_fractional_t_min_never_passes_table(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 1, T_min=150.5, T_max=3000)
        assert T[-1] <= 1000.0

    def test_no_overlap_raises(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="does not overlap"):
                hm.get_tips_table("H2O", 1, T_min=1500, T_max=3000)

    def test_falls_back_to_other_version(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            T, _ = hm.get_tips_table("H2O", 2, T_min=150, T_max=3000, version=2021)
        assert T[-1] == 3000.0
        assert tips_hapi.partitionSum.call_args.kwargs["version"] == 2025

    def test_no_version_has_table_raises(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="No TIPS table"):
                hm.get_tips_table("H2O", 3)

    def test_non_positive_q_raises(self, tips_hapi):
        tips_hapi.partitionSum.side_effect = lambda M, I, T, version=2021: 0.0
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            with pytest.raises(hm.HITRANDataError, match="non-positive"):
                hm.get_tips_table("H2O", 1, T_min=150, T_max=400)

    def test_get_tips_falls_back_too(self, tips_hapi):
        with patch.object(hm, "_get_hapi", return_value=tips_hapi):
            assert hm.get_tips("H2O", 2, 296.0, version=2021) == 296.0
        assert tips_hapi.partitionSum.call_args.kwargs["version"] == 2025


# ── TIPS con el HAPI real (offline: las tablas vienen dentro de HAPI) ─────────

def _real_hapi_available() -> bool:
    try:
        hm._get_hapi()
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _real_hapi_available(), reason="hitran-api not installed")
class TestTipsRealHapi:
    """The isotopologues that broke from_hapi in the 8–14 µm download test."""

    @pytest.mark.parametrize("mol,iso,t_max", [
        ("O3", 1, 1000.0), ("NO2", 1, 1000.0), ("CH4", 1, 2500.0),
    ])
    def test_default_from_hapi_range_works(self, mol, iso, t_max):
        T, Q = hm.get_tips_table(mol, iso, T_min=150.0, T_max=3000.0)
        assert T[0] == 150.0 and T[-1] == t_max
        assert np.all(np.isfinite(Q)) and np.all(Q > 0)

    @pytest.mark.parametrize("mol,iso", [("SO", 2), ("NF3", 1)])
    def test_missing_tips2021_uses_another_version(self, mol, iso):
        T, Q = hm.get_tips_table(mol, iso, T_min=150.0, T_max=400.0)
        assert np.all(np.isfinite(Q)) and np.all(Q > 0)
