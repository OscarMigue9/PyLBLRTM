# Final IASI retrieval

This standalone checkout contains the audited source, small pinned inputs,
tests and validation records for the final IASI retrieval. The forward and
retrieval packages are independently installable under `packages/`.

This directory contains one selected scientific target: `unroll714_sew` with the
z24/Ncol state, 714–1250 cm⁻¹ IASI channels and five retrieved gases. The
recovered step-1,000 checkpoint reproduces the archived T4 results for 300
simulated and 64 ocean scenes through the separately installed packages. The
three declared metric tolerances pass on both sets, and the ocean states agree
scene by scene to `5.77e-6` RMSE per component. The earlier checkpoint copy is
step 400. Numerical parity and historical file identity are tracked separately
in [the baseline gate](docs/baseline_gate.md). The ordered acceptance gates are
in the [reproduction plan](docs/REPRODUCTION_PLAN.md).

## Layout

| Path | Purpose |
|---|---|
| `packages/photonix` | Independently installable clean CPU forward and explicitly versioned historical Torch kernel |
| `packages/iasi_retrieval` | Only the selected z24/Ncol APAE, prior and shared-step retrieval |
| `pipelines` | Verified asset handling, candidate data generation, training, evaluation and Modal runner |
| `configs/final.json` | Frozen physical/model configuration and seeds |
| `manifests` | SHA256 asset, raw-input, cross-section block and provenance inventories |
| `frozen` | Small row selections, continuum inputs and exact GPU pack arrays |
| `tests` | Local unit and numerical regression checks |
| `docs` | Source audit, physics migration, training/data provenance and validation |

Large LUTs, atmospheric data, radiances and model weights live outside this
repository. `data_cache/`, `runs/` and build outputs are ignored and are not
part of this checkout. Asset sizes and SHA256 hashes are in `manifests/`.

## Local setup and checks

Use Python 3.11. From this directory:

```powershell
python -m pip install -e './packages/photonix[torch]'
python -m pip install -e './packages/iasi_retrieval'
python -m pip install pytest ruff modal
python -m pytest -q
python -m ruff check packages pipelines tests
python -m pipelines.assets list
python -m pipelines.assets --cache data_cache fetch split_30k_seed42 frozen_simulated_rows
```

The bundled files are verified without access to the old workspace. Large
frozen inputs use the Drive remote in `manifests/assets.json`; raw CAMS,
ECOSTRESS and AER inputs have pinned public archive URLs in
[`raw_sources.json`](manifests/raw_sources.json). `assets fetch` checks size and
SHA256 before placing a file in the cache. Existing incorrect files cause an
error and are not silently replaced.

## Rebuild from raw sources

The raw downloader fetches only six CAMS NetCDF members, the 1,419 selected
ECOSTRESS spectra and AER spectroscopy. AER plus the bundled TIPS and four
continuum tables rebuilds all three historical GPU pack files **byte for
byte**. The atmospheric states and emissivity derived from those inputs were
also rebuilt with their archived file SHA256 values. See
[`RAW_FETCH.md`](pipelines/RAW_FETCH.md), the
[`generation guide`](pipelines/generation/README.md) and
[`DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for commands and limits.

```powershell
python -m pipelines.raw_fetch all
python -m pipelines.assets --cache data_cache --workspace-root data_cache/raw fetch `
  aer_lines aer_coupling
python -m pipelines.generation --cache data_cache build-gpu-pack `
  --output-dir <new-work-directory>/gpu_pack
```

The source recipe has generated pilot LUT blocks for all five gases in
895–905 cm⁻¹. Reconstructed log cross sections agree with the archived blocks
to numerical precision; [`validation`](docs/validation/README.md) has the five
reports. Modal sampled comparisons cover all 63 blocks for O3, CO2, H2O and
N2O. O3 passes the full 635–1265 cm⁻¹ sampled gate. For CO2, H2O
and N2O, every block and the 714–1250 cm⁻¹ model band pass; only the
1265 cm⁻¹ terminal channel exceeds the full-range maximum. The terminal
differences are `5.23e-4`, `2.90e-4` and `8.70e-4` log10 cross section,
respectively, against a `2.5e-4` limit. The full reports for
[O3](docs/validation/lut_full_o3.json),
[CO2](docs/validation/lut_full_co2.json),
[H2O](docs/validation/lut_full_h2o.json) and
[N2O](docs/validation/lut_full_n2o.json) retain those results. CH4's 63 blocks
and receipts were recovered and verified by SHA256 and source snapshot after
Modal stopped. A local NumPy 2.2.5 expansion was compared with the
manifest-pinned Drive reference. Its sampled gate passes, and an exhaustive
20 × 145 × 630,001-cell scan passes every block and the 714–1250 model band.
The full 635–1265 gate still fails: the 1265 terminal-channel RMS is
`3.65e-5`, above `2.5e-5`, although its maximum and the total-array metrics
pass. See the [CH4 local report](docs/validation/lut_full_ch4_local.json).
The other four gases have sampled comparisons only. Candidate LUTs have not
replaced the frozen training LUTs.

## Reproduction stages

The stages below request a Modal A100-80GB and use persistent
`wikithesis-datos` and `wikithesis-salida` volumes. The initial APAE/prior
reports record an NVIDIA A100 80GB PCIe; the current `samtr7` probe reports
an NVIDIA A100-SXM4-80GB. The Modal secret `rclone-wikithesis` must contain a
working `d400` remote as `RCLONE_CONF_B64` or `RCLONE_CONF`. Run from this directory:

```powershell
modal run pipelines/modal_final.py --stage train-apae
modal run pipelines/modal_final.py --stage train-prior --use-trained-upstream
modal run pipelines/modal_final.py --stage train-unrolled --use-trained-upstream
modal run pipelines/modal_final.py --stage evaluate-simulated --use-trained-upstream
modal run pipelines/modal_final.py --stage evaluate-mar64 --use-trained-upstream
```

On 2026-09-27 at about 20:03 UTC, the `meltraslav08` Modal workspace
`ac-wcltPGHZuWysY1tH2xbeL8` exceeded its spend limit. The fresh SEW run
stopped at step 200 of 1,000, and the remote CH4 job stopped during dense LUT
expansion after all 63 blocks were generated. Both jobs report the
workspace-disabled error in their local logs. CH4 expansion and comparison
were later completed locally as described above. That source workspace remains
spend-limited.
The interrupted run's step-200 checkpoint was backed up outside this checkout
as `runs/sew_step200_backup.pt`, SHA256
`bfbf358626426a7ee86b24cbe935f127bedd1c881063490f15065369dfe929d5`.
A read-only copy fetched from `wikithesis-salida` after the interruption had
the same hash. The [interruption receipt](docs/validation/sew_step200_interrupted.json)
also verifies the persisted APAE, prior and latent hashes against their run
reports. All four artifacts were uploaded to `samtr7` under
`wikithesis-salida/final`, downloaded again and checked for identical byte
counts and SHA256; see the [migration receipt](docs/validation/sew_migration_samtr7.json).
The `samtr7` CPU and A100-80GB probes passed. Resume app
`ap-31MuiofGKfRyiIpnmq3UHP` passed all 12 requested asset fetches by SHA256
and produced a verified step-300 checkpoint. Its backup outside this checkout is
`runs/sew_step300_backup.pt`, SHA256
`7082cf94c3395e4e65dd83e8e8f6df6712c1b739776dcd9bcb43bc036a41fca6`.
It records `paso=300`, 16 input hashes and scheduler epoch 300. The attached
app was deliberately stopped after this checkpoint. Detached resume app
`ap-HRMLdXgqKY22UkradkUpLP` initialized from it without the legacy hash
exception. No checkpoint beyond step 300 or new-model evaluation has been
verified yet.

For future checkpoints written with embedded input hashes, an interrupted
`train-unrolled` can resume from its last complete checkpoint on
`wikithesis-salida` with the same upstream choice:

```powershell
modal run pipelines/modal_final.py --stage train-unrolled --use-trained-upstream --resume-sew
```

The trainer restores the step network, AdamW, cosine scheduler, sampled-row
generator, Torch CPU/CUDA RNG states and existing traces. It verifies the
recipe, residual scale and the SHA256 of all 16 training inputs before
continuing. These include the configuration, APAE and prior checkpoints, prior
latents, five LUTs, states, `dates_30k`, radiances, noise, row split,
`model_error_sf` (`s_f`) and `r_self`. The dates and `s_f` are explicit frozen
assets in [`assets.json`](manifests/assets.json); neither is silently
regenerated during training. Their archived SHA256 values are
`aa1b8c246384fc76baad7e1df7b0b7f38600189f94c5ed397a313af9a0924604`
and `119bca7e97e980d4b41041dca621efd7811d795b0172bb9176454b4e4d801df2`,
respectively. Checkpoints are replaced
atomically every 100 updates and at the final update. Modal commits mounted
Volume changes in the background; an abrupt failure can lose the most recent
uncommitted update, so inspect the recorded `paso` before resuming. Record the
exact GPU subtype when comparing runs: the migration changes PCIe to SXM4,
and byte-identical optimizer trajectories are not guaranteed. The declared
metric tolerances remain the acceptance gate.

A checkpoint from before input hashes were embedded is rejected by default.
After independently verifying its exact file SHA256 and checking the available
job log and upstream run records, use the explicit legacy exception:

```powershell
modal run pipelines/modal_final.py --stage train-unrolled --use-trained-upstream `
  --resume-sew --legacy-checkpoint-sha256 <verified-checkpoint-sha256>
```

The report marks this exception as `legacy_resume_without_embedded_input_hashes`.
It does not retroactively prove which input files produced that old checkpoint.
The step-200 backup above predates embedded input hashes. The `samtr7` Volume
copy and upstream artifacts passed the migration hash checks before the first
resume app started. The step-300 checkpoint embeds all 16 input hashes, so
resuming from it needs no legacy exception. The earlier step-100 backup is
superseded by the step-200 and step-300 copies.

The interrupted Modal SEW training launched on 2026-09-27 used a source snapshot from
before atomic checkpoint writes and resume support were added. The update
formula, sampling and optimizer schedule were unchanged. Its recorded source
hashes therefore differ from this workspace even when the numerical training
trajectory is the same; keep that snapshot with the run record.

Each run fetches and verifies its inputs, writes stage outputs to the run
volume, and records input/output hashes, a complete source snapshot, runtime
versions and metrics. Without
`--use-trained-upstream`, the prior and retrieval stages use the archived
checkpoints listed in `manifests/assets.json`. The same stages can be called
locally as `python -m pipelines.final --help`, but full retrieval needs the
dense LUTs and substantial GPU memory.

Candidate reconstruction of the spectral LUT and radiance inputs is described
in [the generation README](pipelines/generation/README.md). It starts from
manifested raw or intermediate files and checks archived canaries. It must
pass the full physical and model gates before generated data can replace the
frozen training inputs.

The historical 64-scene reference is bundled as
[`frozen/mar_sew_n4.npz`](frozen/mar_sew_n4.npz), independently of the original
workspace. [`mar64_compare`](pipelines/mar64_compare.py) checks a downloaded
Modal scene dump against its SHA256-pinned baseline and reports scene-level
differences.

## Verification status and limits

- Simulated and ocean T4 numerical comparisons pass the fixed tolerances with
  the recovered step-1,000 checkpoint. Its identity with the file used for the
  old evaluation is inferred from scene outputs because that evaluation did
  not record a weight hash.
- The former `sew` path on Drive contains an intermediate step-400 checkpoint.
  It remains in the provenance gate only and is not the final model asset.
- The APAE training run reproduced all 30 historical weight tensors exactly.
  The original prior checkpoint does not record a seed or optimizer recipe;
  its current training configuration is an explicit reconstruction. See
  [`TRAINING_PROVENANCE.md`](docs/TRAINING_PROVENANCE.md).
- The old APAE T/U/D input is frozen separately from the angle-dependent
  radiance/T/U/D path. Its upstream generator is not accredited. The original
  645–2760 cm⁻¹ five-angle radiance parent was downloaded and independently
  SHA256-verified. The archived selection and crop reproduce all 30,000 ×
  2,145 frozen radiance values exactly, with zero differences. This establishes
  the parent-to-frozen data chain, while the physical generation of the parent
  and its five-angle T/U/D remains unaccredited. See the
  [full radiance report](docs/validation/radiance_parent_full.json). A fresh CPU
  noise build
  differs from the frozen 30,000 × 2,145 array in only two float32 cells
  (RMS `1.16e-13`, maximum `9.31e-10`), but is not exact. See
  [`DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) and the
  [full noise report](docs/validation/noise_rebuild.json).
- The clean CPU Photonix recipe and historical dense Torch recipe have
  different physical contracts. The installed historical Torch path has
  full-scene parity; the corrected CPU recipe is not an interchangeable
  replacement for the trained model LUT.
- The strict historical identity gate remains open because the original
  training checkpoint and complete input hashes were never recorded.
