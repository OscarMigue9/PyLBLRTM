"""Immutable molecule definitions and the explicitly supported HITRAN catalogue.

Masses and partition-file identities were checked against HITRAN's isotope
metadata, https://hitran.org/docs/iso-meta/. See resources/tips/manifest.json
for the provenance of the bundled partition-function snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import Iterable


def normalize_isotope_id(value: str | int) -> str:
    """Return the one-character HITRAN ID; decimal 10/11/12 mean 0/A/B."""
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError("isotope_id must be a HITRAN string or integer identifier")
    identifier = str(value).strip().upper()
    identifier = {"10": "0", "11": "A", "12": "B"}.get(identifier, identifier)
    if identifier not in tuple("1234567890AB"):
        raise ValueError(f"Unsupported HITRAN isotope identifier: {value!r}")
    return identifier


@dataclass(frozen=True, slots=True)
class Isotopologue:
    """One isotopologue; partition_file=None requests its bundled Q(T) table."""

    isotope_id: str
    molar_mass_g_mol: float
    partition_file: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "isotope_id", normalize_isotope_id(self.isotope_id))
        mass = float(self.molar_mass_g_mol)
        if not isfinite(mass) or mass <= 0:
            raise ValueError("molar_mass_g_mol must be finite and positive")
        object.__setattr__(self, "molar_mass_g_mol", mass)
        if self.partition_file is not None:
            object.__setattr__(self, "partition_file", Path(self.partition_file))


@dataclass(frozen=True, slots=True)
class Species:
    """A molecule and a nonempty selection of isotopologues, without layer state."""

    name: str
    molecule_id: int
    isotopologues: tuple[Isotopologue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("name must be a nonempty molecule name without outer whitespace")
        if (
            isinstance(self.molecule_id, bool)
            or not isinstance(self.molecule_id, Integral)
            or not 1 <= self.molecule_id <= 99
        ):
            raise ValueError("molecule_id must be an integer from 1 to 99")
        variants = tuple(self.isotopologues)
        if not variants or any(not isinstance(item, Isotopologue) for item in variants):
            raise ValueError("isotopologues must contain at least one Isotopologue")
        identifiers = [item.isotope_id for item in variants]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate isotopologues for {self.name}")
        known = _CATALOGUE.get(self.name)
        if known is not None:
            known_molecule, known_variants = known
            if self.molecule_id != known_molecule:
                raise ValueError(f"{self.name} has HITRAN molecule_id={known_molecule}")
            allowed = {identifier for identifier, _, _ in known_variants}
            if not set(identifiers) <= allowed:
                raise ValueError(f"Unsupported isotope selection for {self.name}: {identifiers}")
        object.__setattr__(self, "molecule_id", int(self.molecule_id))
        object.__setattr__(self, "isotopologues", variants)


# Each tuple is (local ASCII isotope ID, molar mass [g/mol], global TIPS ID).
# CO2 local 9/0/A correspond to global 121/15/120, NOT 15/120/121.
_CATALOGUE = {
    "H2O": (
        1,
        (
            ("1", 18.010565, 1),
            ("2", 20.014811, 2),
            ("3", 19.014780, 3),
            ("4", 19.016740, 4),
            ("5", 21.020985, 5),
            ("6", 20.020956, 6),
            ("7", 20.022915, 129),
        ),
    ),
    "CO2": (
        2,
        (
            ("1", 43.989830, 7),
            ("2", 44.993185, 8),
            ("3", 45.994076, 9),
            ("4", 44.994045, 10),
            ("5", 46.997431, 11),
            ("6", 45.997400, 12),
            ("7", 47.998320, 13),
            ("8", 46.998291, 14),
            ("9", 45.998262, 121),
            ("0", 49.001675, 15),
            ("A", 48.001646, 120),
            ("B", 47.001618, 122),
        ),
    ),
    "O3": (
        3,
        (
            ("1", 47.984745, 16),
            ("2", 49.988991, 17),
            ("3", 49.988991, 18),
            ("4", 48.988960, 19),
            ("5", 48.988960, 20),
        ),
    ),
    "N2O": (
        4,
        (
            ("1", 44.001062, 21),
            ("2", 44.998096, 22),
            ("3", 44.998096, 23),
            ("4", 46.005308, 24),
            ("5", 45.005278, 25),
        ),
    ),
    "CO": (
        5,
        (
            ("1", 27.994915, 26),
            ("2", 28.998270, 27),
            ("3", 29.999161, 28),
            ("4", 28.999130, 29),
            ("5", 31.002516, 30),
            ("6", 30.002485, 31),
        ),
    ),
    "CH4": (
        6,
        (("1", 16.031300, 32), ("2", 17.034655, 33), ("3", 17.037475, 34), ("4", 18.040830, 35)),
    ),
    "O2": (7, (("1", 31.989830, 36), ("2", 33.994076, 37), ("3", 32.994045, 38))),
}


def default_species(
    names: Iterable[str] | str | None = None,
    *,
    isotopologues: Iterable[tuple[int, str | int]] | None = None,
    tips_dir: str | Path | None = None,
) -> tuple[Species, ...]:
    """Select catalogue species, optionally filtering explicit (molecule, isotope) pairs.

    With names=None a pair selection also chooses its molecules. With explicit
    names, every requested molecule must have a selected isotope. No selection
    means all 42 supported isotopologues; it is not an implicit primary-isotope
    approximation. Construct Species directly for custom spectroscopy.
    """
    selected = None
    if isotopologues is not None:
        if isinstance(isotopologues, bool):
            raise ValueError("isotopologues requires explicit (molecule_id, isotope_id) pairs")
        selected = set()
        supported = {(mol, iso) for mol, variants in _CATALOGUE.values() for iso, _, _ in variants}
        for pair in isotopologues:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Each isotope selection must be a (molecule_id, isotope_id) pair")
            molecule, isotope = pair
            if isinstance(molecule, bool) or not isinstance(molecule, Integral):
                raise ValueError("Selected molecule_id must be an integer")
            normalized = (int(molecule), normalize_isotope_id(isotope))
            if normalized not in supported:
                raise ValueError(f"Unknown catalogue isotope selection: {pair!r}")
            if normalized in selected:
                raise ValueError(f"Duplicate isotope selection: {pair!r}")
            selected.add(normalized)
        if not selected:
            raise ValueError("isotopologues selection cannot be empty")
    if names is None:
        requested = [
            name
            for name, (mol, _) in _CATALOGUE.items()
            if selected is None or any(pair[0] == mol for pair in selected)
        ]
    else:
        requested = [names] if isinstance(names, str) else list(names)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("names must be nonempty and contain no duplicates")
    unknown = set(requested) - _CATALOGUE.keys()
    if unknown:
        raise ValueError(f"Unknown catalogue species: {sorted(unknown)}")
    requested_ids = {_CATALOGUE[name][0] for name in requested}
    if selected is not None and any(mol not in requested_ids for mol, _ in selected):
        raise ValueError("Isotope selection includes a molecule outside names")
    base = (
        Path(tips_dir)
        if tips_dir is not None
        else Path(str(files("photonix").joinpath("resources", "tips")))
    )
    result = []
    for name in requested:
        molecule, definitions = _CATALOGUE[name]
        variants = tuple(
            Isotopologue(iso, mass, base / name / f"q{global_id}.txt")
            for iso, mass, global_id in definitions
            if selected is None or (molecule, iso) in selected
        )
        if not variants:
            raise ValueError(f"No isotopologues selected for requested molecule {name}")
        result.append(Species(name, molecule, variants))
    return tuple(result)
