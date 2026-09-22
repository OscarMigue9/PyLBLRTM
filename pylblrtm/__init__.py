"""
pylblrtm — Line-by-line radiative transfer library.

Quick start:
    from pylblrtm import from_hapi, from_files

    data = from_hapi('H2O', nu_min=700, nu_max=1400)
    data = from_files('my_H2O.par', 'my_H2O_tips.txt')
"""

from .data_loader import from_hapi, from_files
from .spectral_data import SpectralData
from .hitran_manager import (
    save_credentials,
    load_credentials,
    cache_status,
    list_cached,
    is_cached,
    list_molecules,
    MOLECULE_IDS,
    DEFAULT_CACHE_DIR,
    HITRANConnectionError,
    HITRANCredentialError,
    HITRANDataError,
)

__version__ = "0.1.0"

__all__ = [
    # Carga de datos
    "from_hapi",
    "from_files",
    "SpectralData",
    # Utilidades HITRAN
    "save_credentials",
    "load_credentials",
    "cache_status",
    "list_cached",
    "is_cached",
    "list_molecules",
    "MOLECULE_IDS",
    "DEFAULT_CACHE_DIR",
    # Excepciones
    "HITRANConnectionError",
    "HITRANCredentialError",
    "HITRANDataError",
]
