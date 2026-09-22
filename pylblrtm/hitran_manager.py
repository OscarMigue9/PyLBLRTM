"""
hitran_manager.py
=================

Basic usage:
    from hitran_manager import get_line_parameters, save_credentials

    # Save credentials once per machine (only needed if HITRAN requires auth)
    save_credentials("your_email@example.com", "your_access_token")

    # Download (or use cache) and return spectral line parameter arrays
    data = get_line_parameters("H2O", nu_min=700, nu_max=1400)
    data["nu0"]    # line centers in cm⁻¹
    data["Sref"]   # line intensities at 296 K

Data license:
    Cite: Gordon et al. 2022, J. Quant. Spectrosc. Radiat. Transfer 277, 107949.
"""

# ─── Imports───────────────────────────────────────────────────────
import numpy as np
import json
import os
import socket
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__) #Saber de dónde viene el log, en este caso hitran_manager.py
import importlib.util as _iutil # Utilidades para cargar módulos
import site as _site #Informacion de los paquetes instalados en el sistema
import sys as _sys #Para manejar el path de los módulos y paquetes


# ─── Importar HAPI desde site-packages ───────────────────────────────────────
def _load_hapi_from_sitepackages():
    # Si ya está cargado y no es el archivo local, reutilizarlo
    if "hapi" in _sys.modules:
        _mod = _sys.modules["hapi"]
        _init = getattr(_mod, "__file__", "") or ""
        # El paquete real vive en site-packages/hapi/__init__.py
        if "site-packages" in _init.replace("\\", "/"):
            return _mod
        # Es el archivo local — descartarlo y buscar el paquete real
        del _sys.modules["hapi"]

    _sp_dirs = (
        _site.getsitepackages()
        if hasattr(_site, "getsitepackages")
        else []
    )
    _sp_dirs.append(_site.getusersitepackages())

    for _sp in _sp_dirs:
        _pkg_init = os.path.join(_sp, "hapi", "__init__.py")
        if os.path.isfile(_pkg_init):
            _spec = _iutil.spec_from_file_location(
                "hapi", _pkg_init,
                submodule_search_locations=[os.path.join(_sp, "hapi")],
            )
            _mod = _iutil.module_from_spec(_spec)
            _sys.modules["hapi"] = _mod
            _spec.loader.exec_module(_mod)
            return _mod

    raise ImportError(
        "\nHAPI (hitran-api) not found in site-packages.\n"
        "Install it with:  pip install hitran-api\n"
        "Verify with:      python -m pip show hitran-api"
    )


_hapi = None  # cargado solo cuando se llama una función que lo necesita

def _get_hapi():
    """Load HAPI from site-packages on first call (lazy import)."""
    global _hapi
    if _hapi is None:
        _hapi = _load_hapi_from_sitepackages()
    return _hapi


# ─── Rutas por defecto ───────────────────────────────────────────────────────
_LIB_HOME = Path.home() / ".pylblrtm"
DEFAULT_CACHE_DIR: Path = _LIB_HOME / "hitran_cache"
_CREDENTIALS_FILE: Path = _LIB_HOME / "credentials.json"

# ─── Catálogo de moléculas HITRAN ─────────────────────────────────────────────
# Referencia completa: https://hitran.org/docs/iso-meta/
MOLECULE_IDS: Dict[str, int] = {
    "H2O": 1,  "CO2": 2,  "O3": 3,   "N2O": 4,  "CO": 5,
    "CH4": 6,  "O2": 7,   "NO": 8,   "SO2": 9,  "NO2": 10,
    "NH3": 11, "HNO3": 12,"OH": 13,  "HF": 14,  "HCl": 15,
    "HBr": 16, "HI": 17,  "ClO": 18, "OCS": 19, "H2CO": 20,
    "HOCl": 21,"N2": 22,  "HCN": 23, "CH3Cl": 24,"H2O2": 25,
    "C2H2": 26,"C2H6": 27,"PH3": 28, "COF2": 29,"SF6": 30,
    "H2S": 31, "HCOOH": 32,"HO2": 33,"O": 34,   "ClONO2": 35,
    "NO+": 36, "HOBr": 37,"C2H4": 38,"CH3OH": 39,"CH3Br": 40,
    "CH3CN": 41,"CF4": 42,"C4H2": 43,"HC3N": 44,"H2": 45,
    "CS": 46,  "SO3": 47, "C2N2": 48,"COCl2": 49,"SO": 51,
    "CH3F": 52,"GeH4": 53,"CS2": 54, "CH3I": 55,"NF3": 56,
}

# Fórmula del isotopólogo 1 (el más abundante) para las moléculas principales
_ISO1_FORMULA: Dict[int, str] = {
    1: "H2(16O)", 2: "(12C)(16O2)", 3: "(16O3)",   4: "(14N2)(16O)",
    5: "(12C)(16O)", 6: "(12C)H4",  7: "(16O2)",   8: "(14N)(16O)",
    9: "(32S)(16O2)", 10: "(14N)(16O2)",
}


# ─── Excepciones personalizadas ───────────────────────────────────────────────

class HITRANConnectionError(OSError):
    """Cannot reach hitran.org (no internet or server down)."""

class HITRANCredentialError(PermissionError):
    """Missing or invalid HITRAN credentials."""

class HITRANDataError(ValueError):
    """HITRAN returned empty or unexpected data."""


# ─── Gestión de credenciales ─────────────────────────────────────────────────

def save_credentials(username: str, access_token: str) -> None:
    """
    Save HITRAN credentials to ~/.pylblrtm/credentials.json.

    The access token can be found at: hitran.org → Log in → Profile → API access.
    Only needs to be called once per machine.

    Parameters
    ----------
    username : str
        Email address registered at hitran.org.
    access_token : str
        Personal API access token from HITRAN.
    """
    _LIB_HOME.mkdir(parents=True, exist_ok=True)
    payload = {"username": username, "access_token": access_token}
    _CREDENTIALS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Permisos solo para el propietario (no aplica en Windows, pero sí en Linux/Mac)
    try:
        os.chmod(_CREDENTIALS_FILE, 0o600)
    except NotImplementedError:
        pass
    logger.info("Credentials saved to %s", _CREDENTIALS_FILE)
    print(f"Credentials saved to {_CREDENTIALS_FILE}")


def load_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Load HITRAN credentials in priority order:
      1. Environment variables: HITRAN_USERNAME and HITRAN_ACCESS_TOKEN
      2. File ~/.pylblrtm/credentials.json
      3. (None, None) — anonymous download (works with HAPI 1.3)

    Returns
    -------
    tuple of (username, access_token) — either may be None
    """
    username = os.environ.get("HITRAN_USERNAME")
    token = os.environ.get("HITRAN_ACCESS_TOKEN")
    if username and token:
        logger.debug("Credentials loaded from environment variables")
        return username, token
    if _CREDENTIALS_FILE.exists():
        cred = json.loads(_CREDENTIALS_FILE.read_text(encoding="utf-8"))
        logger.debug("Credentials loaded from %s", _CREDENTIALS_FILE)
        return cred.get("username"), cred.get("access_token")
    logger.debug("No credentials found")
    return None, None


# ─── Gestión de caché ────────────────────────────────────────────────────────

_METADATA_FILE = "pylblrtm_metadata.json"

def _table_name(mol_name: str, iso_id: int, nu_min: float, nu_max: float) -> str:
    """Deterministic table name for a given download request."""
    return f"{mol_name}_iso{iso_id}_{int(nu_min)}_{int(nu_max)}"


def _write_metadata(table: str, cache_dir: Path, mol_name: str, mol_id: int,
                    iso_id: int, nu_min: float, nu_max: float) -> None:
    """Save download timestamp and parameters to pylblrtm_metadata.json."""
    from datetime import datetime, timezone
    meta_path = cache_dir / _METADATA_FILE
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta[table] = {
        "molecule": mol_name,
        "mol_id": mol_id,
        "iso_id": iso_id,
        "nu_min": nu_min,
        "nu_max": nu_max,
        "source": "hitran.org (current edition)",
        "downloaded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hapi_version": getattr(_get_hapi(), "HAPI_VERSION", "unknown"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_metadata(table: str, cache_dir: Path) -> Optional[Dict]:
    """Read download metadata for a given table."""
    meta_path = cache_dir / _METADATA_FILE
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get(table)


def is_cached(table_name: str, cache_dir: Path) -> bool:
    """Return True if both dataset files (.data and .header) exist on disk."""
    return (
        (cache_dir / f"{table_name}.data").exists()
        and (cache_dir / f"{table_name}.header").exists()
    )


def list_cached(cache_dir: Optional[Path] = None) -> List[str]:
    """Return a sorted list of dataset names downloaded in cache_dir."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    return sorted(f.stem for f in cache_dir.glob("*.data"))


def cache_status(cache_dir: Optional[Path] = None) -> None:
    """Print a summary of cached datasets with download timestamps."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    tables = list_cached(cache_dir)
    if not tables:
        print(f"Cache is empty: {cache_dir}")
        return
    print(f"Datasets in {cache_dir}:\n")
    print(f"  {'Table':<45}  {'KB':>8}  Downloaded (UTC)")
    print("  " + "-" * 80)
    total = 0
    for t in tables:
        size = (cache_dir / f"{t}.data").stat().st_size
        total += size
        meta = _read_metadata(t, cache_dir)
        date_str = meta["downloaded_at"] if meta else "(no metadata)"
        print(f"  {t:<45}  {size/1024:>8.1f}  {date_str}")
    print("  " + "-" * 80)
    print(f"  {'TOTAL':<45}  {total/1024:>8.1f}")
    print()
    print("NOTE: 'hitran.org (current edition)' = HITRAN active at download date.")
    print("      Use force_download=True to fetch the latest version.")


# ─── Verificación de conectividad ────────────────────────────────────────────

def _can_reach_hitran(timeout: int = 5) -> bool:
    try:
        socket.create_connection(("hitran.org", 80), timeout=timeout)
        return True
    except OSError:
        return False


# ─── Descarga y extracción ───────────────────────────────────────────────────

def _extract_line_data(
    table: str, mol_id: int, iso_id: int
) -> Dict[str, np.ndarray]:
    """
    Extract HAPI columns into numpy arrays with physical names.

    Returned columns and their physical meaning:
      nu0    : line center wavenumber (cm⁻¹)
      Sref   : line intensity at 296 K (cm⁻¹/(molec·cm⁻²))
      g_air  : air-broadened Lorentz half-width at 1 atm, 296 K (cm⁻¹/atm)
      g_self : self-broadened Lorentz half-width (cm⁻¹/atm)
      Elow   : lower state energy (cm⁻¹)
      n_air  : temperature exponent for gamma_air
      shift  : air pressure-induced line shift at 1 atm (cm⁻¹/atm)
    """
    params = ["nu", "sw", "gamma_air", "gamma_self", "elower", "n_air", "delta_air"]
    try:
        cols = _get_hapi().getColumns(table, params)
    except Exception as exc:
        raise HITRANDataError(
            f"Could not read table '{table}'. "
            f"Try force_download=True to re-download. Error: {exc}"
        ) from exc

    n = len(cols[0])
    if n == 0:
        raise HITRANDataError(
            f"Table '{table}' is empty. "
            "Check the wavenumber range (there may be no lines in that range)."
        )

    return {
        "nu0":    np.asarray(cols[0], dtype=np.float64),
        "Sref":   np.asarray(cols[1], dtype=np.float64),
        "g_air":  np.asarray(cols[2], dtype=np.float64),
        "g_self": np.asarray(cols[3], dtype=np.float64),
        "Elow":   np.asarray(cols[4], dtype=np.float64),
        "n_air":  np.asarray(cols[5], dtype=np.float64),
        "shift":  np.asarray(cols[6], dtype=np.float64),
        "mol_id": np.full(n, mol_id,  dtype=np.int32),
        "iso_id": np.full(n, iso_id,  dtype=np.int32),
    }


def _classify_error(exc: Exception, mol_name: str) -> None:
    """Convert HAPI exceptions into user-friendly errors."""
    msg = str(exc).lower()
    exc_type = type(exc).__name__.lower()

    if any(k in exc_type for k in ("connectionerror", "urlerror", "socketerror")):
        raise HITRANConnectionError(
            "Could not connect to hitran.org. Check your internet connection."
        ) from exc

    if any(k in msg for k in ("401", "403", "unauthorized", "forbidden")):
        raise HITRANCredentialError(
            "HITRAN rejected the credentials.\n"
            "  1. Check your token at: https://hitran.org/profile/\n"
            "  2. Save them with: hitran_manager.save_credentials('email', 'token')\n"
            "  3. Or set: HITRAN_USERNAME=... HITRAN_ACCESS_TOKEN=... in the environment"
        ) from exc

    if "timeout" in msg:
        raise HITRANConnectionError(
            "HITRAN took too long to respond. The server may be busy.\n"
            "Wait a few minutes and try again."
        ) from exc

    if any(k in msg for k in ("404", "not found")):
        raise HITRANDataError(
            f"HITRAN returned 404 for '{mol_name}'. "
            "Check that the molecule name and wavenumber range are valid."
        ) from exc

    raise HITRANDataError(
        f"Unexpected error downloading '{mol_name}': {type(exc).__name__}: {exc}"
    ) from exc


# ─── API principal ────────────────────────────────────────────────────────────

def get_line_parameters(
    molecule: str,
    nu_min: float,
    nu_max: float,
    iso_id: int = 1,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Fetch HITRAN line parameters, using the local cache when available.

    Checks the local cache before contacting HITRAN. If the data already
    exists on disk, it is loaded without any network request.

    Parameters
    ----------
    molecule : str
        Molecule name, e.g. 'H2O', 'CO2', 'CH4', 'O3'.
        Call list_molecules() to see all available molecules.
    nu_min, nu_max : float
        Spectral range in cm⁻¹.
    iso_id : int
        Isotopologue ID (1 = most abundant). Default: 1.
    cache_dir : Path, optional
        Cache directory. Default: ~/.pylblrtm/hitran_cache.
    force_download : bool
        If True, re-download even if cache exists. Useful to update to a
        newer HITRAN edition.

    Returns
    -------
    dict of numpy arrays:
        nu0, Sref, g_air, g_self, Elow, n_air, shift, mol_id, iso_id

    Raises
    ------
    ValueError
        If the molecule name is not in the HITRAN catalogue.
    HITRANConnectionError
        If there is no internet and no local cache exists.
    HITRANCredentialError
        If HITRAN requires authentication and the credentials are invalid.
    HITRANDataError
        If the requested range has no lines or the data is corrupted.

    Examples
    --------
    >>> data = get_line_parameters("H2O", 700, 1400)
    >>> data = get_line_parameters("CO2", 700, 1400, iso_id=2)
    >>> data = get_line_parameters("CH4", 1200, 1400, cache_dir=Path("/tmp/hapi"))
    """
    mol_name = molecule.upper().strip()
    if mol_name not in MOLECULE_IDS:
        raise ValueError(
            f"Unknown molecule: '{molecule}'.\n"
            f"Call list_molecules() to see all available molecules."
        )
    mol_id = MOLECULE_IDS[mol_name]

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    table = _table_name(mol_name, iso_id, nu_min, nu_max)

    _get_hapi().db_begin(str(cache_dir))

    # ── Caché hit ────────────────────────────────────────────────────────────
    if is_cached(table, cache_dir) and not force_download:
        logger.info("[cache] %s — no download needed", table)
        return _extract_line_data(table, mol_id, iso_id)

    # ── Verificar conectividad antes de intentar descarga ────────────────────
    if not _can_reach_hitran():
        if is_cached(table, cache_dir):
            logger.warning("No internet — using existing cache for %s", table)
            return _extract_line_data(table, mol_id, iso_id)
        raise HITRANConnectionError(
            f"No connection to hitran.org and '{table}' is not in the local cache.\n"
            f"Cache at: {cache_dir}\n"
            "Options:\n"
            "  · Connect to the internet and try again\n"
            "  · Provide a pre-downloaded cache with the cache_dir parameter"
        )

    # ── Descarga ─────────────────────────────────────────────────────────────
    # HAPI 1.3 fetch() signature: (TableName, M, I, numin, numax, ParameterGroups, Parameters)
    # No acepta Username/AccessToken — la descarga es anónima en esta versión.
    # Las credenciales quedan guardadas para cuando HITRAN las requiera (futuras versiones).
    logger.info("[download] %s  %.1f–%.1f cm⁻¹  iso=%d", mol_name, nu_min, nu_max, iso_id)
    try:
        _get_hapi().fetch(table, mol_id, iso_id, nu_min, nu_max)
        _write_metadata(table, cache_dir, mol_name, mol_id, iso_id, nu_min, nu_max)

    except (HITRANConnectionError, HITRANCredentialError, HITRANDataError):
        raise
    except Exception as exc:
        _classify_error(exc, mol_name)

    return _extract_line_data(table, mol_id, iso_id)


# ─── Funciones de partición (TIPS) ───────────────────────────────────────────
# HAPI 1.3 incluye TIPS-2011/2017/2021/2025 embebidas.
# No se necesita descargar archivos .txt adicionales.
# Referencia: Gamache et al. 2021, JQSRT 271, 107713.

_TIPS_VERSIONS = (2025, 2021, 2017, 2011)


def get_partition_sum(
    molecule: str,
    iso_id: int,
    T: float,
    version: int = 2021,
) -> float:
    """
    Return Q(T) — the total internal partition sum at temperature T.

    Uses the TIPS tables embedded in HAPI (no additional download required).
    Required to scale line intensities with temperature:
        S(T) = S_ref * Q(T_ref)/Q(T) * exp(...) * (1 - exp(...))

    Parameters
    ----------
    molecule : str
        Molecule name, e.g. 'H2O', 'CO2'.
    iso_id : int
        Isotopologue ID (1 = most abundant).
    T : float
        Temperature in Kelvin.
    version : int
        TIPS version to use: 2011, 2017, 2021, or 2025. Default: 2021.

    Returns
    -------
    float
        Dimensionless Q(T).

    Examples
    --------
    >>> Q296 = get_partition_sum('H2O', 1, 296.0)
    >>> Q300 = get_partition_sum('H2O', 1, 300.0)
    >>> ratio = Q296 / Q300   # partition ratio for scaling S_ref
    """
    mol_name = molecule.upper().strip()
    if mol_name not in MOLECULE_IDS:
        raise ValueError(f"Unknown molecule: '{molecule}'.")
    mol_id = MOLECULE_IDS[mol_name]

    if version not in _TIPS_VERSIONS:
        raise ValueError(f"version must be one of {_TIPS_VERSIONS}.")

    try:
        return float(_get_hapi().partitionSum(mol_id, iso_id, T, version=version))
    except Exception as exc:
        raise HITRANDataError(
            f"Could not compute Q(T={T} K) for {molecule} iso {iso_id}.\n"
            f"Check that the molecule has TIPS data for version {version}. Error: {exc}"
        ) from exc


def get_partition_sum_table(
    molecule: str,
    iso_id: int,
    T_min: float = 1.0,
    T_max: float = 3000.0,
    version: int = 2021,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the full Q(T) table over [T_min, T_max] at 1 K steps.

    Useful for interpolating Q(T) instead of calling get_partition_sum()
    at every temperature step.

    Parameters
    ----------
    molecule : str
        Molecule name.
    iso_id : int
        Isotopologue ID.
    T_min, T_max : float
        Temperature range in Kelvin.
    version : int
        TIPS version: 2011, 2017, 2021, or 2025.

    Returns
    -------
    T_arr : np.ndarray
        Temperature array in Kelvin, 1 K step.
    Q_arr : np.ndarray
        Corresponding Q(T) values.

    Examples
    --------
    >>> T, Q = get_partition_sum_table('H2O', 1, T_min=150, T_max=400)
    >>> Q_at_280 = float(np.interp(280.0, T, Q))
    """
    mol_name = molecule.upper().strip()
    if mol_name not in MOLECULE_IDS:
        raise ValueError(f"Unknown molecule: '{molecule}'.")
    mol_id = MOLECULE_IDS[mol_name]

    if version not in _TIPS_VERSIONS:
        raise ValueError(f"version must be one of {_TIPS_VERSIONS}.")

    T_arr = np.arange(max(1.0, T_min), min(T_max, 9999.0) + 1.0, 1.0)
    try:
        Q_arr = np.array([
            _get_hapi().partitionSum(mol_id, iso_id, float(t), version=version)
            for t in T_arr
        ], dtype=np.float64)
    except Exception as exc:
        raise HITRANDataError(
            f"Error computing Q(T) table for {molecule} iso {iso_id}: {exc}"
        ) from exc

    return T_arr, Q_arr


# ─── Conversión HAPI ↔ formatos estándar HITRAN ──────────────────────────────
#
# Formato .par  (HITRAN-2004, 160 chars/línea):
#   cols  1- 2   mol_id         %2d
#   col   3      iso_id         %1d
#   cols  4-15   nu             %12.6f
#   cols 16-25   sw             %10.3E
#   cols 26-35   a              %10.3E   (Einstein A-coeff)
#   cols 36-40   gamma_air      %5.4f
#   cols 41-45   gamma_self     %5.3f
#   cols 46-55   elower         %10.4f
#   cols 56-59   n_air          %4.2f
#   cols 60-67   delta_air      %8.6f
#   cols 68-82   global_upper   %15s
#   cols 83-97   global_lower   %15s
#   cols 98-112  local_upper    %15s
#   cols113-127  local_lower    %15s
#   cols128-133  ierr           %6s
#   cols134-145  iref           %12s
#   col 146      flag           %1s
#   cols147-153  g_upper        %7.1f
#   cols154-160  g_lower        %7.1f
#
# Formato TIPS .txt (HITRANonline):
#   Dos columnas separadas por espacios: T(K)   Q(T)
#   Una fila por kelvin entero, sin cabecera.


def hapi_to_par(
    molecule: str,
    nu_min: float,
    nu_max: float,
    output_path: "str | Path",
    iso_id: int = 1,
    cache_dir: Optional[Path] = None,
    force_download: bool = False,
) -> Path:
    """
    Download HAPI data and write it as a standard HITRAN-2004 .par file.

    The output file is identical to what you would download manually from
    HITRANonline → Download → .par, and can be loaded with par_to_dict().

    Parameters
    ----------
    molecule : str
        Molecule name, e.g. 'H2O'.
    nu_min, nu_max : float
        Spectral range in cm⁻¹.
    output_path : str or Path
        Path of the .par file to write.
    iso_id : int
        Isotopologue ID. Default: 1.
    cache_dir : Path, optional
        Cache directory. Default: ~/.pylblrtm/hitran_cache.
    force_download : bool
        Re-download even if cache exists.

    Returns
    -------
    Path
        Path of the written .par file.
    """
    mol_name = molecule.upper().strip()
    if mol_name not in MOLECULE_IDS:
        raise ValueError(f"Unknown molecule: '{molecule}'.")
    mol_id = MOLECULE_IDS[mol_name]

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    table = _table_name(mol_name, iso_id, nu_min, nu_max)
    _get_hapi().db_begin(str(cache_dir))

    if not is_cached(table, cache_dir) or force_download:
        if not _can_reach_hitran():
            raise HITRANConnectionError("No connection to hitran.org.")
        _get_hapi().fetch(table, mol_id, iso_id, nu_min, nu_max)
        _write_metadata(table, cache_dir, mol_name, mol_id, iso_id, nu_min, nu_max)

    # Extraer todos los campos necesarios para el .par completo
    all_params = [
        "nu", "sw", "a", "gamma_air", "gamma_self", "elower", "n_air",
        "delta_air", "global_upper_quanta", "global_lower_quanta",
        "local_upper_quanta", "local_lower_quanta",
        "ierr", "iref", "line_mixing_flag", "gp", "gpp",
    ]
    cols = _get_hapi().getColumns(table, all_params)
    (nu, sw, a, g_air, g_self, elow, n_air, d_air,
     gu, gl, lu, ll, ierr, iref, flag, gp, gpp) = cols

    def _f5_4(v: float) -> str:
        # Fortran F5.4: 5 chars, 4 decimales.
        # 0 <= v < 1 → '.xxxx'.  v >= 1 → trunca a 5.
        s = f"{v:.4f}"
        if s.startswith("0."):
            return s[1:]
        return s[:5]

    def _f4_2(v: float) -> str:
        # Fortran F4.2: 4 chars, 2 decimales (n_air puede ser negativo).
        # -1 < v < 0 → '-.xx'.  resto → trunca a 4.
        s = f"{v:.2f}"
        if s.startswith("-0."):
            return "-" + s[2:]   # "-.xx" — 4 chars
        return s[:4]

    def _f8_6(v: float) -> str:
        # Fortran F8.6: 8 chars, 6 decimales.
        # -1 < v < 0 → '-.xxxxxx'.  resto → trunca a 8.
        s = f"{v:.6f}"
        if s.startswith("-0."):
            return "-" + s[2:]   # "-.xxxxxx" — 8 chars
        return s[:8]

    n = len(nu)
    lines = []
    for i in range(n):
        gu_s   = str(gu[i])[:15].ljust(15)
        gl_s   = str(gl[i])[:15].ljust(15)
        lu_s   = str(lu[i])[:15].ljust(15)
        ll_s   = str(ll[i])[:15].ljust(15)
        ierr_s = str(ierr[i])[:6].ljust(6)
        iref_s = str(iref[i])[:12].ljust(12)
        flag_s = str(flag[i])[:1].ljust(1)

        try:
            gp_s  = f"{float(gp[i]):7.1f}"
            gpp_s = f"{float(gpp[i]):7.1f}"
        except (ValueError, TypeError):
            gp_s  = "    0.0"
            gpp_s = "    0.0"

        line = (
            f"{mol_id:2d}{iso_id:1d}"
            f"{nu[i]:12.6f}"
            f"{sw[i]:10.3E}"
            f"{a[i]:10.3E}"
            f"{_f5_4(g_air[i])}"
            f"{g_self[i]:5.3f}"
            f"{elow[i]:10.4f}"
            f"{_f4_2(n_air[i])}"
            f"{_f8_6(d_air[i])}"
            f"{gu_s}{gl_s}{lu_s}{ll_s}"
            f"{ierr_s}{iref_s}{flag_s}"
            f"{gp_s}{gpp_s}"
        )
        lines.append(line)

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %d lines to %s", n, output_path)
    return output_path


def par_to_dict(par_path: "str | Path") -> Dict[str, np.ndarray]:
    """
    Read a standard HITRAN-2004 .par file and return a dict of numpy arrays.

    Returns the same dict format as get_line_parameters(), compatible with
    the forward model. Accepts files downloaded from HITRANonline or
    generated by hapi_to_par().

    Parameters
    ----------
    par_path : str or Path
        Path to the .par file.

    Returns
    -------
    dict of numpy arrays:
        nu0, Sref, g_air, g_self, Elow, n_air, shift, mol_id, iso_id
    """
    par_path = Path(par_path)
    if not par_path.exists():
        raise FileNotFoundError(f".par file not found: {par_path}")

    nu0, Sref, g_air, g_self, Elow, n_air, shift = [], [], [], [], [], [], []
    mol_ids, iso_ids = [], []

    with par_path.open(encoding="utf-8", errors="replace") as f:
        for line_num, raw in enumerate(f, 1):
            if len(raw.rstrip("\n")) < 155:   # líneas incompletas al final
                continue
            try:
                mol_ids.append(int(raw[0:2]))
                iso_ids.append(int(raw[2:3]))
                nu0.append(float(raw[3:15]))
                Sref.append(float(raw[15:25]))
                # raw[25:35] = Einstein A (no se usa en el dict estándar)
                g_air.append(float(raw[35:40]))
                g_self.append(float(raw[40:45]))
                Elow.append(float(raw[45:55]))
                n_air.append(float(raw[55:59]))
                shift.append(float(raw[59:67]))
            except ValueError as exc:
                logger.warning("Malformed line %d in %s: %s", line_num, par_path.name, exc)

    if not nu0:
        raise HITRANDataError(f"No valid lines read from {par_path}.")

    return {
        "nu0":    np.asarray(nu0,     dtype=np.float64),
        "Sref":   np.asarray(Sref,    dtype=np.float64),
        "g_air":  np.asarray(g_air,   dtype=np.float64),
        "g_self": np.asarray(g_self,  dtype=np.float64),
        "Elow":   np.asarray(Elow,    dtype=np.float64),
        "n_air":  np.asarray(n_air,   dtype=np.float64),
        "shift":  np.asarray(shift,   dtype=np.float64),
        "mol_id": np.asarray(mol_ids, dtype=np.int32),
        "iso_id": np.asarray(iso_ids, dtype=np.int32),
    }


def tips_to_txt(
    molecule: str,
    iso_id: int,
    output_path: "str | Path",
    T_min: float = 1.0,
    T_max: float = 3000.0,
    version: int = 2021,
) -> Path:
    """
    Write the TIPS table embedded in HAPI as a standard two-column .txt file.

    The output format (T  Q, one row per integer Kelvin, no header) is
    identical to the file downloaded from HITRANonline, and can be read
    back with txt_to_partition_table().

    Parameters
    ----------
    molecule : str
        Molecule name.
    iso_id : int
        Isotopologue ID.
    output_path : str or Path
        Path of the .txt file to write.
    T_min, T_max : float
        Temperature range in Kelvin (1 K step).
    version : int
        TIPS version: 2011, 2017, 2021, or 2025.

    Returns
    -------
    Path
        Path of the written .txt file.
    """
    T_arr, Q_arr = get_partition_sum_table(molecule, iso_id, T_min, T_max, version)
    output_path = Path(output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for T, Q in zip(T_arr, Q_arr):
            f.write(f"{T:8.1f}  {Q:14.6f}\n")
    logger.info("TIPS written to %s (%d points)", output_path, len(T_arr))
    return output_path


def txt_to_partition_table(txt_path: "str | Path") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a TIPS .txt file (two columns: T and Q) and return numpy arrays.

    Accepts files downloaded from HITRANonline or generated by tips_to_txt().
    Blank lines and lines starting with '#' are ignored.

    Parameters
    ----------
    txt_path : str or Path
        Path to the TIPS file.

    Returns
    -------
    T_arr : np.ndarray
        Temperatures in Kelvin.
    Q_arr : np.ndarray
        Corresponding Q(T) values.

    Examples
    --------
    >>> T, Q = txt_to_partition_table("H2O_tips.txt")
    >>> Q_at_280 = float(np.interp(280.0, T, Q))
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"TIPS file not found: {txt_path}")

    T_list, Q_list = [], []
    with txt_path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            if len(parts) < 2:
                continue
            try:
                T_list.append(float(parts[0]))
                Q_list.append(float(parts[1]))
            except ValueError:
                continue

    if not T_list:
        raise HITRANDataError(f"No data read from {txt_path}.")

    return np.asarray(T_list, dtype=np.float64), np.asarray(Q_list, dtype=np.float64)


def list_molecules() -> None:
    """Print the catalogue of molecules available in HITRAN."""
    print(f"\n{'ID':>4}  {'Name':<12}  Principal isotopologue")
    print("-" * 50)
    for name, mid in sorted(MOLECULE_IDS.items(), key=lambda x: x[1]):
        formula = _ISO1_FORMULA.get(mid, "see hitran.org/docs/iso-meta/")
        print(f"{mid:>4}  {name:<12}  {formula}")
    print()
