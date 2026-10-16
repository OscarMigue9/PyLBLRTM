# Photonix

Photonix is the independently installable forward library for the final IASI
retrieval. Version `0.2.0` has two explicit numerical contracts:

- `photonix` computes thermal, clear-sky nadir radiance from AER line data or a
  manifest-backed SVD cross-section LUT on the 714–1250 cm⁻¹ band. Its recipe is
  `aer-3.8.1-convf4-thermal-714-1250-v2`.
- `photonix.torch_legacy` preserves the differentiable dense-LUT kernels used by
  the selected z24/Ncol checkpoint. Its recipe is
  `retrieval-z24-ncol-torch-frozen-v1`. It requires the optional Torch extra.

These contracts have different Planck and H₂O continuum constants, LUT formats,
and boundary behavior. They are not interchangeable for a trained checkpoint.
See [the migration record](../../docs/FORWARD_MIGRATION.md) for provenance and
validation limits.

## Install

From this directory, with Python 3.10 or newer:

```powershell
python -m pip install -e .
python -m pip install -e '.[torch]'  # also install the historical Torch backend
```

The CPU runtime depends on NumPy, SciPy and Numba. Importing `photonix` does not
require Torch. Install the `torch` extra when importing `photonix.torch_legacy` or
running the final retrieval. For a standalone wheel, run `python -m pip wheel
--no-deps .` here.

## Public API

| Import | Purpose |
|---|---|
| `AtmosphericProfile`, `Layer`, `Species`, `Isotopologue`, `default_species` | Explicit atmosphere, hydrostatic layers and isotope selection |
| `LineList`, `read_hitran_par`, `fetch_hitran_par`, `AerLineMixing` | Line inputs and optional AER CO₂ coupling |
| `CrossSectionLUT` | Manifest-validated SVD log-cross-section reader |
| `run_nadir_simulation`, `NadirForwardOutput` | Vertical optical depth, transmittance, upwelling and downwelling |
| `at_sensor_radiance`, `at_sensor_radiance_components`, `InstrumentLineShape` | Surface composition and optional Gaussian channel convolution |
| `planck_radiance_nu`, `brightness_temperature_nu`, `voigt_profile` | Individual radiative and spectroscopy kernels |

`photonix.torch_legacy` exports `interp_sigma`, `integrate_tud`, `planck_nu`,
`prep_continuum_tables`, `continuum_od_torch` and the complete `forward_lut`.
The retrieval package constructs its dense LUT axes, gas order, state and channel
windows; these are deliberately not implicit in the library.

### Minimal CPU example

The AER line file is an external input. Select its record format explicitly:

```python
from photonix import (
    AtmosphericProfile,
    at_sensor_radiance,
    default_species,
    run_nadir_simulation,
)

profile = AtmosphericProfile(
    pressure_hpa=[100.0, 1000.0],
    temperature_k=[230.0, 290.0],
    vmr={"H2O": [0.001, 0.01]},
)
water = default_species(["H2O"], isotopologues=[(1, "1")])
atmosphere = run_nadir_simulation(
    profile,
    water,
    parfile="/path/to/aer_lines.par",
    record_format="aer100",
    nu_min_cm1=899.0,
    nu_max_cm1=901.0,
    step_cm1=0.01,
)
radiance_mw = at_sensor_radiance(atmosphere, skin_temperature_k=295.0, epsilon=0.98)
```

The profile is ordered from top of atmosphere to surface. Wavenumber is in
cm⁻¹; returned radiance is mW m⁻² sr⁻¹ (cm⁻¹)⁻¹. `run_nadir_simulation` accepts
exactly one of `parfile`, `lines` or `lut`. A real SVD LUT must provide an explicit
manifest identifying its physical recipe, isotopologues, continuum convention
and block files. No scientific data are fetched during import or by default.

## Bundled resources and provenance

The wheel contains 42 local TIPS partition tables, a compact H₂O/CO₂ AER
continuum table, their JSON manifests and the distinct historical H₂O continuum
table required by `torch_legacy`. External line lists, CO₂ coupling data,
atmospheres, LUT blocks, training data and model weights are not included.
The TIPS manifest records per-file SHA256 values and explicitly states that the
original local files lack TIPS edition metadata. The continuum manifests record
source hashes and the two runtime conventions. See `src/photonix/resources/` and
`src/photonix/torch_legacy/continuum_tables.npz`.

This package has **no declared software license**. The acquisition history and
redistribution terms of the bundled scientific tables are also incomplete.
Establish those rights before publishing the package outside the thesis
workspace; the data should be cited to their original AER/HITRAN/TIPS sources.

## Compatibility and verification

The CPU API and `torch_legacy` recipe are versioned separately. Future changes
to a public signature require a package version update. A change to a numerical
constant, interpolation, continuum or transfer formula also requires a new
recipe identifier and parity checks before a stored checkpoint can use it.
Version `0.2.0` does not promise numerical equivalence between the two backends.

The workspace tests are in `../../tests/forward/`. From the workspace root:

```powershell
python -m pytest -q tests/forward
python -m ruff check packages/photonix tests/forward
python -m ruff format --check packages/photonix tests/forward
```

The real-LUT smoke test needs external blocks and an explicit manifest. The
software tests and small historical Torch fixtures do not certify full-band
spectroscopy or checkpoint accuracy; those gates are tracked in the workspace
validation reports.
