"""
pylblrtm — Line-by-line radiative transfer library.

Quick start — forward model:
    from pylblrtm import from_hapi
    from pylblrtm.atmosphere import AtmosphericProfile
    from pylblrtm.forward import compute_transmittance
    from pylblrtm.profiles import us_standard_atmosphere

    sd      = from_hapi("CO2", 700, 750, iso_id=1)
    profile = us_standard_atmosphere(gases=["CO2"])
    nu, T   = compute_transmittance(sd, profile, 710, 740)

Quick start — data layer only:
    data = from_hapi('H2O', nu_min=700, nu_max=1400)
    data = from_files('my_H2O.par', 'my_H2O_tips.txt')
"""

from .data_loader import from_hapi, from_files
from .spectral_data import SpectralData
from .atmosphere import AtmosphericProfile, HomogeneousPath
from .forward import compute_transmittance
from .profiles import us_standard_atmosphere, list_profile_gases
from .config import run_forward, run_path, load_config, ForwardResult
from .binning import bin_by_wavelength
from .continuum import continuum_optical_depth, ContinuumTables
from .hitran_manager import (
    save_credentials,
    load_credentials,
    get_molar_mass,
    get_abundance,
    cache_status,
    list_cached,
    is_cached,
    list_molecules,
    MOLECULE_IDS,
    DEFAULT_CACHE_DIR,
    HITRANConnectionError,
    HITRANQuotaError,
    HITRANCredentialError,
    HITRANDataError,
    HITRANNoLinesError,
    HITRANNotAvailableError,
)

__version__ = "0.1.2"

__all__ = [
    # Datos espectrales
    "from_hapi",
    "from_files",
    "SpectralData",
    # Perfil atmosférico
    "AtmosphericProfile",
    "HomogeneousPath",
    # Forward model (directo)
    "compute_transmittance",
    # Forward model (YAML-driven)
    "run_forward",
    "run_path",
    "bin_by_wavelength",
    "continuum_optical_depth",
    "ContinuumTables",
    "load_config",
    "ForwardResult",
    # Perfiles de referencia
    "us_standard_atmosphere",
    "list_profile_gases",
    # Utilidades HITRAN
    "save_credentials",
    "load_credentials",
    "get_molar_mass",
    "get_abundance",
    "cache_status",
    "list_cached",
    "is_cached",
    "list_molecules",
    "MOLECULE_IDS",
    "DEFAULT_CACHE_DIR",
    # Excepciones
    "HITRANConnectionError",
    "HITRANQuotaError",
    "HITRANCredentialError",
    "HITRANDataError",
    "HITRANNoLinesError",
    "HITRANNotAvailableError",
]
