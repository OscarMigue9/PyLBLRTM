# Candidate regeneration pipeline

This pipeline regenerates the final 714–1250 cm⁻¹ training inputs from frozen
assets. Its physics follows the first compatible Photonix source snapshot,
`9f93c955779c11f7a8018e9200d63dddfa22ba23` (2026-08-19). The archived LUT
was generated on 2026-08-14 and radiance metadata on 2026-08-17. Neither artifact
records a source commit. Outputs therefore have **candidate** status until the
numerical canaries and the baseline gate establish parity.

Run commands from the `thesis final` directory. `pipelines.assets` verifies each
asset against `manifests/assets.json`; this pipeline checks cached assets again at
the point of use. It never downloads or launches Modal work implicitly.

## Raw atmospheric and emissivity inputs

`manifests/raw_catalogs.json` records the size and SHA256 of each local raw file.
Only six of the twenty CAMS NetCDF subsets supply the 30,000 profiles, in the
fixed order `temp, hum, co2, oz, n2o, ch4`. The twenty-first directory entry is
the dataset PDF. `build-states` verifies the six source files before extraction.
The L137 pressure coefficients and the May 2026 extraction arithmetic are pinned
in `l137.py` and `raw.py`. The full rebuild produced the archived SHA256
`9fd71327df0b3f65ed03ef35f466ff4ec85c2c3498c7b58a3689084a6658859e`.

`build-emissivity` verifies `record_ids.npy` and each of the 1419 selected
ECOSTRESS spectra. Their ordered rows are interpolated to the original 645–2760
cm⁻¹ channel grid. The full rebuild produced the archived SHA256
`a33e89a55138ecf46d763a1d1aa6dc4074b29d73c0ca0f9c4d791bdb7d2129e1`.
These byte canaries were run locally against the frozen outputs; they establish
identity of those two derived assets, conditional on obtaining the catalogued
raw bytes on another machine.

```powershell
python -m pipelines.raw_fetch cams
python -m pipelines.raw_fetch ecostress
python -m pipelines.assets --cache data_cache --workspace-root data_cache/raw fetch `
  emissivity_record_ids states_30k emissivity_iasi
python -m pipelines.generation build-states --workspace-root data_cache/raw `
  --output <work>/nwpsaf_30k_states.npz `
  --archive data_cache/data/nwpsaf_30k_states.npz
python -m pipelines.generation --cache data_cache build-emissivity --workspace-root data_cache/raw `
  --output <work>/emis_iasi.npy `
  --archive data_cache/inputs/emissivity/emis_iasi.npy
```

Use `--per-subset 2` and `--count 2` for small numeric canaries. Full output
is never overwritten.

## Historical spectroscopy pack

The AER archive in `manifests/raw_sources.json` regenerates the 730,759-line
IASI PAR and `lncpl_lines` byte-identically. The final Photonix package bundles
all 42 TIPS Q(T) files, each pinned in its own resource manifest. Four small
continuum tables, totalling 66,491 bytes, are bundled under `frozen/continuum/`
and pinned in `manifests/gpu_pack_sources.json`. The H₂O table was independently
rebuilt byte-identically from the local `absco-ref_wv-mt-ckd.nc` snapshot; the
exact original Fortran source for the CO₂, O₂/N₂ and bandhead tables is absent,
so those three tables remain frozen scientific inputs.

```powershell
python -m pipelines.raw_fetch aer
python -m pipelines.assets --cache data_cache --workspace-root data_cache/raw fetch `
  aer_lines aer_coupling
python -m pipelines.generation --cache data_cache build-gpu-pack `
  --output-dir <work>/gpu_pack
```

`build-gpu-pack` verifies every input before writing and then checks the SHA256
of all three outputs against `manifests/assets.json`. The local reconstruction
matched `lineas.npz`, `qtab.npz` and `cont.npz` byte for byte. It preserves the
**historical** CO₂ Q(T) mapping: local isotope 9 uses q15, local 10 uses q120,
and local A uses q121. The corrected Photonix CPU catalogue uses q121, q15 and
q120 for isotopes 9, 0 and A respectively. The archived pack must retain the
historical mapping for LUT/model parity. The edition and acquisition record of
the original TIPS files are unknown; the frozen files themselves are verified.

## SVD LUT and synthesis

```powershell
python -m pipelines.assets --cache data_cache --workspace-root data_cache/raw fetch `
  gpu_pack_lines gpu_pack_qtab gpu_pack_continuum states_30k `
  emissivity_iasi water_refractive_index r_self
python -m pipelines.generation --cache data_cache preflight
```

The SVD archives are an external input because the final asset manifest records
only the dense five-gas caches. Supply their directory explicitly. It must contain
`lut_<GAS>_<LOW>_<HIGH>.npz` blocks and `r_self.npz`. The canary checks seven
archived hashes from the 895–905 pilot and complete, continuous 635–1265 coverage.
It establishes storage identity for those files; it does not validate the physics.

```powershell
python -m pipelines.generation canary --blocks-dir <svd-block-directory>
python -m pipelines.generation --cache data_cache build-rself `
  --output <work>/r_self.npz
python -m pipelines.generation --cache data_cache build-lut `
  --gas CO2 --start 895 --end 905 --output-dir <work>/lut
python -m pipelines.generation canary --blocks-dir <svd-block-directory> `
  --candidate <work>/lut/lut_CO2_895_905.npz `
  --archived <svd-block-directory>/lut_CO2_895_905.npz
```

`build-lut` requires CuPy and a CUDA 12 GPU. The historical candidate environment
used NumPy 2.1.3, SciPy 1.14.1 and `cupy-cuda12x` 13.3.0. It writes one 10 cm⁻¹
block at a time with temperature-major rows, 20 × 145 T/P states and rank 64.
The CUDA line kernel is copied from the pinned snapshot without comments. Its
historical pressure-shift behavior is preserved deliberately for the canary.
The builder supports 635–1265, including the guard needed by the dense model LUT.
The full 645–2760 experiment and its χ/bandhead terms are outside this pipeline.

```powershell
python -m pipelines.generation expand-lut --blocks-dir <svd-block-directory> `
  --output-dir <work>/dense --gas H2O --start 635 --end 1265
python -m pipelines.generation --cache data_cache build-tud `
  --blocks-dir <svd-block-directory> --output-dir <work>/tud_chunks --count 2
python -m pipelines.generation --cache data_cache assemble `
  --chunks-dir <work>/tud_chunks --count 2 `
  --radiance-output <work>/radiances_iasiang.npz `
  --tud-output <work>/tud_iasiang.npy
python -m pipelines.generation build-noise --count 2 --output <work>/noise_iasiang.npz
```

Remove `--count 2` for the 30,000-profile run. T/U/D requires CUDA; the other
steps use NumPy/SciPy. The T/U/D chunks are resumable and retain all five viewing
angles. `assemble` applies angle seed 991 and land emissivity seed 42 to the same
profile order, and writes matching angle T/U/D and radiance. `build-noise` first
generates 645–1250 noise using seed 1337 and batch size 256, then crops channel
276 onward to preserve the historical realization pattern. Changing the batch
size changes the generated noise and must be treated as a new run.

`assemble` and `build-noise` accept `--archive <verified-frozen-file>` to compare
numerical arrays for the generated profile prefix. Their output notes explicitly
mark new candidate generation, so numerical parity is reported separately from
whole-file SHA identity. The archived 714–1250 radiance has also been checked
directly against four complete rows of its original 645–2760 five-angle parent
on Drive, without downloading that 5.08 GB file. See
[`radiance_parent.json`](../../manifests/radiance_parent.json), the
[`range verifier`](../radiance_parent_canary.py) and the
[`provenance report`](../../docs/DATA_PROVENANCE.md). This establishes exact
selected-row identity, not complete radiance regeneration from the SVD LUT.

The complete 30,000-profile CPU noise build was compared independently with
[`noise_compare.py`](../noise_compare.py), which validates the frozen SHA256,
the generation report's candidate SHA256, shapes, channel axis and all
64,350,000 samples. Two float32 noise cells differ (RMS `1.16e-13`), while
`nu` and `sigma_ch` match exactly. The candidate is therefore not byte or
array identical; see the [full report](../../docs/validation/noise_rebuild.json).

The archived `tud_iasi_all.npy` is an older nadir APAE input. A T/U/D generated
from these August SVD factors is a **new candidate** and must not overwrite that
asset. Place all outputs in a new work directory. Do not label them identical to
the frozen training inputs based only on shape or a passing storage canary.

## Full five-gas candidate LUT on Modal

`pipelines/modal_lut_full.py` runs the same `build-lut` and `expand-lut` commands
as above, one model gas at a time. Each run schedules all 63 aligned 10 cm^-1
blocks from 635 to 1265 cm^-1, then expands them to one dense float32 array.
Its A100 block calls and CPU expansion calls use the manifest-pinned
`lineas.npz`, `qtab.npz` and `cont.npz` from `frozen/gpu_pack/`, materialized
through `pipelines.assets` into the `wikithesis-datos` volume. The existing
`rclone-wikithesis` secret is configured for any manifest entry with a remote
source. Every pack file is SHA256 checked before each stage.

From `thesis final`, select a Modal profile with access to the
`rclone-wikithesis` secret and the `wikithesis-datos` volume:

```powershell
$env:MODAL_PROFILE = '<your-profile>'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
modal run pipelines/modal_lut_full.py --gas H2O
modal run pipelines/modal_lut_full.py --gas CO2
modal run pipelines/modal_lut_full.py --gas O3
modal run pipelines/modal_lut_full.py --gas N2O
modal run pipelines/modal_lut_full.py --gas CH4
```

Each gas writes `GAS/blocks/lut_GAS_LOW_HIGH.npz` and a JSON receipt for each
block, then `GAS/dense/lut_band645_GAS.npy` and `GAS/report.json` to the
`thesis-final-lut-candidates` Modal volume. Restart the same command to reuse
completed blocks; the runner checks their hashes, source snapshot and pack
inputs before skipping them. It refuses a changed completed artifact rather
than silently mixing code revisions. An interrupted block or dense expansion
without a complete receipt is regenerated. Reports record input, source,
candidate block and dense hashes, plus the archived SHA256 as a reference.

For example, download a completed report with:

```powershell
modal volume get thesis-final-lut-candidates CO2/report.json runs/lut_CO2_report.json
```

After a gas finishes, run the read-only full-band verifier. It checks the
candidate dense LUT and all 63 block/receipt SHA256 values, verifies the
builder's source snapshot and GPU-pack hashes, and fetches the archived dense
LUT from the manifest if the verified cache is absent. The report is written
locally; the candidate volume is never changed by this verifier.

```powershell
modal run pipelines/modal_lut_compare.py --gas CO2 `
  --output docs/validation/lut_full_co2.json
```

The comparison uses 5 temperature indices, 4 pressure indices, 301 channels
within each of the 63 blocks, and the final 1265 cm^-1 channel. Its report
records every index and per-block statistics. Each block and the overall grid
must have RMS log10 cross-section difference at most `2.5e-5` and maximum
absolute difference at most `2.5e-4`, the same limits fixed for the pilot.
Passing this sampled comparison does not establish equality at every LUT cell.

For an exhaustive numerical comparison of the two dense files, use the optional
streaming scan. It reads every float32 value using bounded row/channel chunks,
then records RMS and maximum differences for all 63 blocks, the exact inclusive
714-1250 cm^-1 retrieval band, the final 1265 cm^-1 channel and the complete
array. The fixed pilot limits apply to each reported region. The scan saves a
separate report and retains the sampled comparison in it.

```powershell
modal run pipelines/modal_lut_compare.py --gas CO2 --full-scan `
  --output docs/validation/lut_full_co2_scan.json
```

The five-gas candidate remains incomplete until all 315 blocks and full-band
comparisons are recorded under `docs/validation/`. These outputs remain
candidate LUTs even after generation: numerical comparison against archived
factors and downstream model metrics must be checked before replacing the
frozen dense LUTs. The five 895-905 cm^-1 pilot comparisons passed, but do not
establish full-band parity.

The separate, restartable Modal runner for 714–1250 cm^-1 T/U/D, radiance and
noise candidates is documented in [`../MODAL_DATA_CANDIDATE.md`](../MODAL_DATA_CANDIDATE.md).
