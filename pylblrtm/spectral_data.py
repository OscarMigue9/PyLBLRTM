"""
spectral_data.py — Uniform spectral data container for PyLBLRTM.

Supports one or more (mol_id, iso_id) pairs in a single object.
The forward model calls Q(T, mol_id, iso_id) per line to get the correct
partition sum for each isotopologue.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class SpectralData:
    """
    Holds everything the forward model needs on the data side.

    Attributes
    ----------
    lines : dict
        Line parameter arrays with keys:
          nu0, Sref, g_air, g_self, Elow, n_air, shift, mol_id, iso_id.
        mol_id and iso_id tag every line so the forward model knows which
        isotopologue each line belongs to.
    tips : dict
        Partition sum tables keyed by (mol_id, iso_id).
        Each value is a (T_arr, Q_arr) tuple of 1-D numpy arrays.
    molar_masses : dict
        Molar mass [g/mol] keyed by (mol_id, iso_id).
    molecule : str, optional
        Descriptive label (e.g. 'H2O'). None when the object holds
        lines from more than one molecule.
    """

    lines: Dict[str, np.ndarray]
    tips: Dict[tuple, Tuple[np.ndarray, np.ndarray]]
    molar_masses: Dict[tuple, float]
    molecule: Optional[str] = None

    # ── Partition sum ─────────────────────────────────────────────────────────

    def Q(self, T: float, mol_id: int, iso_id: int) -> float:
        """Partition sum Q(T) for the given (mol_id, iso_id), interpolated."""
        key = (int(mol_id), int(iso_id))
        if key not in self.tips:
            raise KeyError(
                f"No TIPS data for (mol_id={mol_id}, iso_id={iso_id}). "
                f"Available: {sorted(self.tips.keys())}"
            )
        T_arr, Q_arr = self.tips[key]
        # np.interp would silently return the edge value outside the table
        if not (T_arr[0] <= T <= T_arr[-1]):
            raise ValueError(
                f"T={T:.2f} K is outside the TIPS table of (mol_id={mol_id}, "
                f"iso_id={iso_id}): {T_arr[0]:.1f}–{T_arr[-1]:.1f} K"
            )
        return float(np.interp(T, T_arr, Q_arr))

    # ── Quick access ──────────────────────────────────────────────────────────

    @property
    def nu0(self) -> np.ndarray:
        return self.lines["nu0"]

    @property
    def Sref(self) -> np.ndarray:
        return self.lines["Sref"]

    @property
    def n_lines(self) -> int:
        return len(self.lines["nu0"])

    @property
    def mol_ids(self) -> Tuple[int, ...]:
        """HITRAN molecule IDs present, in ascending order."""
        return tuple(sorted({m for m, _ in self.tips}))

    # ── Combining and splitting ───────────────────────────────────────────────

    def for_molecule(self, mol_id: int) -> "SpectralData":
        """The lines, TIPS and molar masses of one molecule only."""
        mask = self.lines["mol_id"] == mol_id
        if not mask.any():
            raise KeyError(f"No lines of mol_id={mol_id}; present: {self.mol_ids}")
        return SpectralData(
            lines={k: v[mask] for k, v in self.lines.items()},
            tips={k: v for k, v in self.tips.items() if k[0] == mol_id},
            molar_masses={k: v for k, v in self.molar_masses.items() if k[0] == mol_id},
            molecule=self.molecule if len(self.mol_ids) == 1 else None,
        )

    @classmethod
    def combine(cls, items: "list[SpectralData]") -> "SpectralData":
        """One SpectralData with the lines of several (e.g. one per gas)."""
        items = list(items)
        if not items:
            raise ValueError("combine() needs at least one SpectralData")
        if len(items) == 1:
            return items[0]
        lines = {k: np.concatenate([sd.lines[k] for sd in items]) for k in items[0].lines}
        tips: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
        masses: Dict[tuple, float] = {}
        for sd in items:
            tips.update(sd.tips)
            masses.update(sd.molar_masses)
        labels = {sd.molecule for sd in items}
        return cls(lines=lines, tips=tips, molar_masses=masses,
                   molecule=labels.pop() if len(labels) == 1 else None)

    def __repr__(self) -> str:
        keys = sorted(self.tips.keys())
        nu = self.lines["nu0"]
        iso_str = ", ".join(f"({m},{i})" for m, i in keys)
        label = self.molecule or "multi-molecule"
        return (
            f"SpectralData({label}, "
            f"(mol_id,iso_id)=[{iso_str}], "
            f"{self.n_lines} lines, "
            f"nu=[{nu.min():.1f}, {nu.max():.1f}] cm-1)"
        )
