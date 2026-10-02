"""
data_loader.py — Two paths to load spectral data into PyLBLRTM.

    from_hapi(molecule, nu_min, nu_max, ...)   → SpectralData
    from_files(par_path, tips, ...)            → SpectralData

Both return a SpectralData that can hold one or many (mol_id, iso_id) pairs.
The forward model calls Q(T, mol_id, iso_id) per line.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np

from . import hitran_manager as hm
from .spectral_data import SpectralData


_ID_TO_MOL: dict[int, str] = {v: k for k, v in hm.MOLECULE_IDS.items()}


# ── API pública ───────────────────────────────────────────────────────────────

def from_hapi(
    molecule: str,
    nu_min: float,
    nu_max: float,
    iso_id: "int | list[int] | str" = 1,
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
    iso_id : int, list of int, or 'all'
        Isotopologue(s) to load.
        - int: single isotopologue (1 = most abundant). Default: 1.
        - list[int]: specific isotopologues, e.g. [1, 2, 3].
        - 'all': every isotopologue available for the molecule.
        TIPS and molar mass are loaded automatically for each isotopologue.
    tips_T_min, tips_T_max : float
        Temperature range for the TIPS tables. Default 150–3000 K.
    tips_version : int
        TIPS version: 2011, 2017, 2021 (default) or 2025.
    **hapi_kwargs
        Extra arguments for get_line_parameters()
        (e.g. cache_dir, force_download).

    Returns
    -------
    SpectralData

    Examples
    --------
    >>> sd = from_hapi("H2O", 700, 1400)
    >>> sd = from_hapi("H2O", 700, 1400, iso_id=[1, 2, 3])
    >>> sd = from_hapi("H2O", 700, 1400, iso_id="all")
    """
    lines = hm.get_line_parameters(molecule, nu_min, nu_max, iso_id=iso_id, **hapi_kwargs)

    # Derive the actual (mol_id, iso_id) pairs from the downloaded lines
    pairs = sorted(set(zip(lines["mol_id"].tolist(), lines["iso_id"].tolist())))

    tips: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
    molar_masses: Dict[tuple, float] = {}
    for mol_id, iso in pairs:
        mol_name = _ID_TO_MOL[mol_id]
        T_arr, Q_arr = hm.get_tips_table(
            mol_name, iso,
            T_min=tips_T_min, T_max=tips_T_max,
            version=tips_version,
        )
        tips[(mol_id, iso)] = (T_arr, Q_arr)
        molar_masses[(mol_id, iso)] = hm.get_molar_mass(mol_name, iso)

    return SpectralData(
        lines=lines,
        tips=tips,
        molar_masses=molar_masses,
        molecule=molecule.upper().strip(),
    )


def from_files(
    par_path: Union[str, Path],
    tips: Dict[tuple, Union[str, Path, Tuple[np.ndarray, np.ndarray]]],
    molar_masses: Optional[Dict[tuple, float]] = None,
) -> SpectralData:
    """
    Load spectral data from a HITRAN .par file and per-isotopologue TIPS.

    Parameters
    ----------
    par_path : str or Path
        HITRAN-2004 .par file. May contain multiple molecules and isotopologues.
    tips : dict
        TIPS data keyed by (mol_id, iso_id). Each value is either:
        - a path (str or Path) to a two-column TIPS .txt file, or
        - a (T_arr, Q_arr) tuple of numpy arrays already in memory.
        Every (mol_id, iso_id) pair present in the .par must have an entry.
    molar_masses : dict, optional
        Molar masses [g/mol] keyed by (mol_id, iso_id).
        If omitted, looked up automatically from HAPI's isotopologue table.

    Returns
    -------
    SpectralData

    Examples
    --------
    >>> sd = from_files(
    ...     "mezcla.par",
    ...     tips={(1, 1): "H2O_iso1.txt", (1, 2): "H2O_iso2.txt"},
    ... )
    >>> sd = from_files(
    ...     "mezcla.par",
    ...     tips={(1, 1): (T_arr, Q_arr)},
    ...     molar_masses={(1, 1): 18.010565},
    ... )
    """
    lines = hm.par_to_dict(par_path)

    present = set(zip(lines["mol_id"].tolist(), lines["iso_id"].tolist()))

    missing = present - set(tips.keys())
    if missing:
        raise ValueError(
            f"Missing TIPS for isotopologue(s): {sorted(missing)}. "
            "Provide a path or (T, Q) arrays for each (mol_id, iso_id) in the .par file."
        )

    # Load TIPS — skip entries not present in the .par
    loaded_tips: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
    for key, val in tips.items():
        if key not in present:
            continue
        if isinstance(val, (str, Path)):
            loaded_tips[key] = hm.txt_to_tips(val)
        else:
            T_arr, Q_arr = val
            loaded_tips[key] = (np.asarray(T_arr, dtype=np.float64),
                                np.asarray(Q_arr, dtype=np.float64))

    # Molar masses
    loaded_masses: Dict[tuple, float] = {}
    if molar_masses is not None:
        loaded_masses = {k: float(v) for k, v in molar_masses.items()}
    else:
        for mol_id, iso in present:
            mol_name = _ID_TO_MOL.get(mol_id)
            if mol_name is None:
                raise ValueError(
                    f"mol_id={mol_id} not in the internal catalogue. "
                    "Pass molar_masses manually."
                )
            loaded_masses[(mol_id, iso)] = hm.get_molar_mass(mol_name, iso)

    # Molecule label — single name if only one molecule, None if mixed
    mol_ids_present = set(k[0] for k in present)
    molecule = _ID_TO_MOL.get(next(iter(mol_ids_present))) if len(mol_ids_present) == 1 else None

    return SpectralData(
        lines=lines,
        tips=loaded_tips,
        molar_masses=loaded_masses,
        molecule=molecule,
    )
