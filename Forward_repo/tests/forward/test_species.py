"""Catalogue selection and molecule metadata regressions."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from photonix.spectroscopy.species import Isotopologue, Species, default_species


def test_explicit_pairs_keep_same_local_id_in_different_molecules_separate():
    selected = default_species(isotopologues=[(1, "2"), (2, "1"), (2, "A")])
    assert [(item.name, [iso.isotope_id for iso in item.isotopologues]) for item in selected] == [
        ("H2O", ["2"]),
        ("CO2", ["1", "A"]),
    ]


def test_complete_catalogue_and_correct_rare_co2_partition_identity():
    catalogue = default_species()
    assert sum(len(item.isotopologues) for item in catalogue) == 42
    co2 = default_species("CO2")[0]
    rare = {item.isotope_id: item for item in co2.isotopologues}
    # HITRAN isotope metadata: global ID is NOT ordered by local ID for CO2.
    assert rare["9"].partition_file.name == "q121.txt"
    assert rare["0"].partition_file.name == "q15.txt"
    assert rare["A"].partition_file.name == "q120.txt"
    assert rare["B"].partition_file.name == "q122.txt"
    assert rare["0"].molar_mass_g_mol == 49.001675
    assert rare["A"].molar_mass_g_mol == 48.001646


def test_decimal_aliases_match_ascii_hitran_selection():
    co2 = default_species(isotopologues=[(2, "10"), (2, "11"), (2, "12")])[0]
    assert [iso.isotope_id for iso in co2.isotopologues] == ["0", "A", "B"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"isotopologues": True},
        {"isotopologues": []},
        {"isotopologues": [(1, "A")]},
        {"isotopologues": [(1, "1"), (1, "1")]},
        {"isotopologues": [(2, "10"), (2, "0")]},
        {"names": ["H2O"], "isotopologues": [(2, "1")]},
        {"names": ["H2O", "CO2"], "isotopologues": [(1, "1")]},
        {"names": ["unknown"]},
        {"names": ["H2O", "H2O"]},
    ],
)
def test_bad_or_ambiguous_selection_raises(kwargs):
    with pytest.raises(ValueError):
        default_species(**kwargs)


def test_species_are_immutable_and_have_no_layer_state(tmp_path):
    gas = default_species("H2O", tips_dir=tmp_path)[0]
    assert isinstance(gas.isotopologues, tuple)
    assert gas.isotopologues[0].partition_file == tmp_path / "H2O" / "q1.txt"
    with pytest.raises(FrozenInstanceError):
        gas.name = "CO2"
    with pytest.raises(FrozenInstanceError):
        gas.isotopologues[0].molar_mass_g_mol = 42.0
    assert not hasattr(gas, "Pmol")
    assert not hasattr(gas, "QT")


@pytest.mark.parametrize("mass", [0, -1, float("nan"), float("inf")])
def test_invalid_mass_rejected(mass):
    with pytest.raises(ValueError):
        Isotopologue("1", mass)


def test_species_rejects_empty_duplicate_and_noninteger_definitions():
    iso = Isotopologue("1", 18.0)
    for mol, variants in [(1, ()), (1, (iso, iso)), (True, (iso,)), (1.5, (iso,))]:
        with pytest.raises(ValueError):
            Species("H2O", mol, variants)


def test_species_rejects_known_molecule_name_or_isotope_mismatch():
    with pytest.raises(ValueError, match="molecule_id=1"):
        Species("H2O", 2, (Isotopologue("1", 18.0),))
    with pytest.raises(ValueError, match="Unsupported isotope"):
        Species("H2O", 1, (Isotopologue("A", 18.0),))


def test_bundled_tables_match_provenance_manifest():
    first = default_species("H2O")[0].isotopologues[0].partition_file
    base = Path(first).parents[1]
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == 42
    for entry in manifest["files"]:
        content = (base / entry["path"]).read_bytes()
        assert len(content) == entry["bytes"]
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
