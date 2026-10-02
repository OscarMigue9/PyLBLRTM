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
