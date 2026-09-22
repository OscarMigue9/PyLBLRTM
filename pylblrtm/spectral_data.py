"""
spectral_data.py — Uniform spectral data container for PyLBLRTM.

The forward model receives a SpectralData object regardless of whether
the data came from user-provided files or from HAPI.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class SpectralData:
    """
    Holds everything the forward model needs on the data side.

    Attributes
    ----------
    molecule : str
        Molecule name (e.g. 'H2O', 'CO2').
    iso_id : int
        HITRAN isotopologue ID (1 = most abundant isotopologue).
    lines : dict
        Numpy arrays of line parameters. Keys:
          nu0, Sref, g_air, g_self, Elow, n_air, shift, mol_id, iso_id
    tips_T : ndarray
        Temperature grid of the TIPS table [K].
    tips_Q : ndarray
        Corresponding partition sums Q(T).
    molar_mass : float
        Molar mass in g/mol. Required for computing the Doppler width.
    """

    molecule: str
    iso_id: int
    lines: Dict[str, np.ndarray]
    tips_T: np.ndarray
    tips_Q: np.ndarray
    molar_mass: float

    # ── Consultas que el forward hace sobre los datos ─────────────────────

    def Q(self, T: float) -> float:
        """Partition sum interpolated to temperature T [K]."""
        return float(np.interp(T, self.tips_T, self.tips_Q))

    # ── Accesos rápidos a los arrays de línea ─────────────────────────────

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
        nu = self.lines["nu0"]
        return (
            f"SpectralData({self.molecule} iso{self.iso_id}, "
            f"{self.n_lines} lines, "
            f"nu=[{nu.min():.1f}, {nu.max():.1f}] cm-1, "
            f"Wg={self.molar_mass} g/mol)"
        )
