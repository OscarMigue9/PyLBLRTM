"""
data_loader.py — Two paths to load spectral data into PyLBLRTM.

    from_files(par_path, tips_txt_path)              → SpectralData
    from_hapi(molecule, nu_min, nu_max, ...)         → SpectralData

Both functions return the same SpectralData object. The forward model
calls one or the other depending on what data is available.
"""

from pathlib import Path
from typing import Optional, Union

from . import hitran_manager as hm
from .spectral_data import SpectralData


# ── Tabla de masas molares (g/mol) ────────────────────────────────────────────
# Isotopólogo principal de cada molécula HITRAN.
# Fuente: HITRAN documentation / NIST.
_MOLAR_MASS: dict[str, float] = {
    "H2O":   18.010565,
    "CO2":   43.989830,
    "O3":    47.984745,
    "N2O":   44.001062,
    "CO":    27.994915,
    "CH4":   16.031300,
    "O2":    31.989830,
    "NO":    29.997989,
    "SO2":   63.961901,
    "NO2":   45.992904,
    "NH3":   17.026549,
    "HNO3":  62.995645,
    "OH":    17.002740,
    "HF":    20.006229,
    "HCl":   35.976678,
    "HBr":   79.926160,
    "HI":   127.912297,
    "OCS":   59.966986,
    "H2CO":  30.010565,
    "HCN":   27.010899,
    "C2H2":  26.015650,
    "C2H6":  30.046950,
    "H2S":   33.987721,
    "N2":    28.006148,
    "PH3":   33.997238,
    "SF6":  145.962492,
    "H2":     2.015650,
    "CS":    43.971940,
    "SO3":   79.956820,
}

# Lookup inverso mol_id → nombre (para autodetección en from_files)
_ID_TO_MOL: dict[int, str] = {v: k for k, v in hm.MOLECULE_IDS.items()}


# ── Helpers internos ──────────────────────────────────────────────────────────

def _resolve_molecule(lines: dict, molecule: Optional[str]) -> str:
    """Return the molecule name, auto-detecting from mol_id if needed."""
    if molecule:
        return molecule
    unique_ids = set(int(x) for x in lines["mol_id"])
    if len(unique_ids) > 1:
        raise ValueError(
            f"The .par file contains {len(unique_ids)} distinct molecules "
            f"(mol_ids={unique_ids}). Pass molecule='NAME' to specify which one to use."
        )
    mol_id = unique_ids.pop()
    name = _ID_TO_MOL.get(mol_id)
    if name is None:
        raise ValueError(
            f"mol_id={mol_id} is not in the internal catalogue. "
            "Pass molecule='NAME' manually."
        )
    return name


def _resolve_molar_mass(molecule: str, molar_mass: Optional[float]) -> float:
    """Return molar mass in g/mol, looking it up in the internal table if not given."""
    if molar_mass is not None:
        return float(molar_mass)
    Wg = _MOLAR_MASS.get(molecule)
    if Wg is None:
        raise ValueError(
            f"Molar mass unknown for '{molecule}'. "
            "Pass molar_mass=<value in g/mol>."
        )
    return Wg


# ── API pública ───────────────────────────────────────────────────────────────

def from_files(
    par_path: Union[str, Path],
    tips_txt_path: Union[str, Path],
    molecule: Optional[str] = None,
    iso_id: int = 1,
    molar_mass: Optional[float] = None,
) -> SpectralData:
    """
    Load spectral data from user-provided files.

    Parameters
    ----------
    par_path : str or Path
        HITRAN-2004 .par file (160 chars/line).
    tips_txt_path : str or Path
        Two-column TIPS file: T(K)  Q(T), no header.
        Downloadable from HITRANonline or generated with hm.tips_to_txt().
    molecule : str, optional
        Molecule name (e.g. 'H2O'). If omitted, auto-detected from the
        mol_id in the .par file (only works if the file has a single molecule).
    iso_id : int
        Isotopologue ID (default 1 = most abundant).
    molar_mass : float, optional
        Molar mass in g/mol. If omitted, looked up in the internal table.

    Returns
    -------
    SpectralData
    """
    lines = hm.par_to_dict(par_path)
    tips_T, tips_Q = hm.txt_to_partition_table(tips_txt_path)

    mol = _resolve_molecule(lines, molecule)
    Wg = _resolve_molar_mass(mol, molar_mass)

    return SpectralData(
        molecule=mol,
        iso_id=iso_id,
        lines=lines,
        tips_T=tips_T,
        tips_Q=tips_Q,
        molar_mass=Wg,
    )


def from_hapi(
    molecule: str,
    nu_min: float,
    nu_max: float,
    iso_id: int = 1,
    molar_mass: Optional[float] = None,
    tips_T_min: float = 150.0,
    tips_T_max: float = 3000.0,
    tips_version: int = 2021,
    **hapi_kwargs,
) -> SpectralData:
    """
    Load spectral data using HAPI (downloads from HITRAN if not cached).

    Parameters
    ----------
    molecule : str
        Molecule name (e.g. 'H2O', 'CO2', 'CH4').
    nu_min, nu_max : float
        Spectral range in cm⁻¹.
    iso_id : int
        Isotopologue ID (default 1 = most abundant).
    molar_mass : float, optional
        Molar mass in g/mol. If omitted, looked up in the internal table.
    tips_T_min, tips_T_max : float
        Temperature range for the TIPS table. Default 150–3000 K.
    tips_version : int
        TIPS version: 2011, 2017, 2021 (default) or 2025.
    **hapi_kwargs
        Additional arguments passed to get_line_parameters()
        (e.g. cache_dir, force_download).

    Returns
    -------
    SpectralData
    """
    lines = hm.get_line_parameters(
        molecule, nu_min, nu_max, iso_id=iso_id, **hapi_kwargs
    )
    tips_T, tips_Q = hm.get_partition_sum_table(
        molecule, iso_id,
        T_min=tips_T_min, T_max=tips_T_max,
        version=tips_version,
    )
    Wg = _resolve_molar_mass(molecule, molar_mass)

    return SpectralData(
        molecule=molecule,
        iso_id=iso_id,
        lines=lines,
        tips_T=tips_T,
        tips_Q=tips_Q,
        molar_mass=Wg,
    )
