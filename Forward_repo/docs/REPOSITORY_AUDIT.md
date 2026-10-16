# Repository audit and final scope

## Decision recorded

The reproducible target is `unroll714_sew` with an 818-component Ncol state,
24 latent variables, five gases and 2,145 IASI channels from 714 to 1250 cm⁻¹.
Its final learned update has three training iterations and four inference
iterations. The target includes the simulated test selection and 64 ocean
scenes. The fixed comparison tolerances are 0.001 dex for mean gas RMSE and
0.05 K each for temperature and skin temperature RMSE. GPU training need not
produce byte-identical weights, but it must use the same verified inputs and
configuration and meet these numerical criteria.

This standalone checkout contains the selected source, pinned small inputs,
tests and validation records. Generated artifacts are excluded. The original
thesis workspace contains three independent Git repositories (`photonix`, `model`, `wiki`), experimental
outputs, writing and figures, operations scripts and raw data. Its root has no
Git history that could identify one final pipeline. The source snapshot and
origin commits for the selected pieces are in
[`code_provenance.json`](../manifests/code_provenance.json). The ordered
acceptance gates are in [`REPRODUCTION_PLAN.md`](REPRODUCTION_PLAN.md).

## Dependency chain

| Stage | Final input/output contract | Current evidence |
|---|---|---|
| Raw atmosphere and surface | Six NWP-SAF CAMS L137 NetCDF members, 1,419 selected ECOSTRESS spectra, AER v3.8.1 | Public downloads and per-file SHA256 in [`raw_sources.json`](../manifests/raw_sources.json) and [`raw_catalogs.json`](../manifests/raw_catalogs.json) |
| Derived atmosphere and emissivity | 30,000 Ncol states, dates, selected ECOSTRESS emissivity rows | States and emissivity rebuilt byte-identically; `dates_30k` is an explicit SHA256-verified frozen asset; fixed row selections bundled in `frozen/` |
| Spectroscopy pack | AER line/coupling files, 42 bundled TIPS tables, four bundled continuum tables, three GPU pack arrays | All three pack NPZs rebuilt byte-identically; see [`gpu_pack_rebuild.json`](validation/gpu_pack_rebuild.json) |
| Spectral LUT | 315 SVD blocks over 635–1265 cm⁻¹ for five gases, five dense log-cross-section arrays, `r_self` | Four gases have 63-block sampled comparisons: O3 passes the sampled full range; CO2, H2O and N2O pass all blocks and the 714–1250 model band but fail the 1265 terminal maximum. CH4's 63 blocks were recovered, locally expanded and exhaustively scanned: blocks and model band pass; terminal RMS fails. |
| Synthetic training data | APAE T/U/D, angle-dependent radiance, noise, `model_error_sf` | The five-angle parent was downloaded and independently hashed; its selected 30,000 × 2,145 radiances match the frozen archive exactly. Full noise comparison found 2 differing float32 values in 64,350,000 cells (RMS `1.16e-13`). `model_error_sf` is a frozen input; physical generation of the five-angle parent and the APAE T/U/D remain unaccredited. See [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md). |
| Learned retrieval | APAE z24 decoder, geographic prior, shared SEW step-1000 network | APAE retraining reproduced all 30 model tensors; seed-7 prior retraining is numerically different from the frozen prior. Fresh SEW reached step 300 after migration to an enabled Modal workspace and detached continuation started from its hash-verified checkpoint. New end-to-end evaluation remains pending. |
| Evaluation of frozen model | Original 300 simulated test rows and 64 named ocean scenes | Extracted T4 runtime with recovered step-1000 checkpoint passes numerical tolerances on both sets. The historical sea scene dump is bundled as [`frozen/mar_sew_n4.npz`](../frozen/mar_sew_n4.npz); strict historical identity remains open. |

The large arrays and weights are deliberately external to the source
distribution. `pipelines.assets` checks size and SHA256 before use. Small
selection files, continuum resources, TIPS text tables and the three GPU pack
arrays are bundled because they are scientific inputs and byte canaries for the
executable recipe. The pack arrays were reconstructed from those inputs and
checked against their archived hashes.

## Source disposition

| Original area | Disposition in this directory |
|---|---|
| `forward clean/photonix/src/photonix` | Clean CPU forward copied into independently installable `packages/photonix` |
| `model/physinv/forward_lut_torch.py` and `continuum_torch.py` | Numerical kernel extracted as the explicitly named `photonix.torch_legacy` backend required by the selected checkpoint |
| Final z24/Ncol scripts under `model/physinv/closure_forward/experimentos/apae_loss_fisica` | APAE, prior, retrieval and forward adapter factored into `packages/iasi_retrieval`; old ad hoc script import paths removed |
| Final LUT/data scripts and source snapshots in `photonix/reference` and `experiments` | Candidate generation commands, raw fetcher and frozen manifests; historical physics differences documented |
| Later z36/VMR, memory-10, hybrid OE and other ablations | Excluded from the final model package and config |
| `wiki`, writing, figures and application code | Excluded because they do not run this physical/model chain |

The corrected CPU forward and the historical dense Torch recipe have distinct
physical contracts. Replacing the historical LUT or kernel with the corrected
CPU recipe without a numerical comparison would change the trained model's
inputs. [`FORWARD_MIGRATION.md`](FORWARD_MIGRATION.md) records those differences.

## Verified results and open gates

The recovered SEW step-1000 checkpoint is the frozen final candidate. The
earlier `unroll714_sew_ckpt.pt` is step 400. The step-1000 checkpoint matches
the archived 300-scene trajectory at float32 precision. Evaluation through the
extracted packages passes all three declared T4 metric tolerances on the
simulated 300 and ocean 64 sets. The ocean states match the historical dump to
`5.77e-6` RMSE per component. The run reports and numerical deltas are in
[`validation`](validation/README.md). These runs validate the **recovered
checkpoint and frozen inputs**, not a model retrained from scratch.

The AER PAR/coupling files were regenerated from the pinned public archive.
Together with the bundled TIPS and continuum inputs, they reproduce
`lineas.npz`, `qtab.npz` and `cont.npz` byte for byte. The historical CO2
partition-table assignment is preserved for this pack; the corrected CPU
forward has a different assignment. The original acquisition of the TIPS
edition and three continuum source tables is not recorded. The five pilot LUT
blocks at 895–905 cm⁻¹ (CO2, H2O, O3, N2O, CH4) each pass the declared
reconstructed log-cross-section RMS and maximum-error limits. Their NPZ hashes
differ from the archive, so this is numerical spectral agreement, not byte
identity. The pilot reports are [CO2](validation/lut_pilot_co2_895_905.json),
[H2O](validation/lut_pilot_h2o_895_905.json),
[O3](validation/lut_pilot_o3_895_905.json),
[N2O](validation/lut_pilot_n2o_895_905.json) and
[CH4](validation/lut_pilot_ch4_895_905.json).

Complete sampled comparisons now cover 63 candidate blocks each for
[O3](validation/lut_full_o3.json), [CO2](validation/lut_full_co2.json),
[H2O](validation/lut_full_h2o.json) and [N2O](validation/lut_full_n2o.json).
O3 passes the declared 635–1265 cm⁻¹ gate. All 63 individual blocks and the
714–1250 cm⁻¹ model band pass for the other three gases, but the full-range
maximum fails at the terminal 1265 cm⁻¹ channel: CO2 `5.23e-4`, H2O
`2.90e-4`, N2O `8.70e-4` log10 cross section against the `2.5e-4` limit.
The candidate dense LUT hashes differ from archived outputs. These four
sampled comparisons do not establish equality at every LUT cell or numerical
parity of freshly synthesized radiances. Full-cell scans of O3, CO2, H2O and
N2O remain undone.

The remote CH4 expansion stopped when Modal disabled the workspace. Its 63
candidate blocks and receipts were then verified by SHA256 and source snapshot,
and expanded locally with NumPy 2.2.5. The local dense SHA256 is
`b6ccc049b20bf9f1f91dc3f327c29727540ec496d54b52d73a34a007872ac187`.
The comparison used the manifest-pinned Drive reference SHA256
`4585dca36e91fee0e124056917e7f657afc01c0f6db053bf6782254dc1b8e9fe`,
not a stale similarly named output-volume file. The
[local report](validation/lut_full_ch4_local.json) passes the sampled gate,
then exhaustively scans all 20 × 145 × 630,001 float32 cells. Every block and
the exact 714–1250 model band pass (band RMS `6.856e-7`, maximum `7.629e-6`).
The total-array RMS `7.151e-7` and maximum `1.793e-4` also pass. The
terminal 1265 cm⁻¹ RMS `3.651e-5` exceeds `2.5e-5`, so the **full-range
exhaustive gate fails** even though its terminal maximum passes. Candidate
CH4 remains separate from the frozen training LUT.

APAE was retrained for 300 epochs from the fixed inputs: all 30 model-state
tensors equal the archived tensors exactly and best validation loss differs by
`6.36e-10`. The retrained seed-7 geographic prior differs from the archived
prior (`0.335` RMSE over the 30,000-by-24 latent array). The original prior
trainer, seed and stopping history were not saved; seed 0 is inferred from its
weight initialization, not certified metadata. Fresh SEW training stopped at
step 200 of 1,000 after the Modal workspace was disabled. A backup outside this
checkout (`runs/sew_step200_backup.pt`) and a read-only copy fetched afterward
from the output volume share SHA256
`bfbf358626426a7ee86b24cbe935f127bedd1c881063490f15065369dfe929d5`.
There is no accepted end-to-end retraining result yet. See
[`TRAINING_PROVENANCE.md`](TRAINING_PROVENANCE.md).

SEW resume checkpoints now embed and compare SHA256 for all 16 inputs:
configuration, states, dates, radiance, noise, split, `s_f`, `r_self`, five
dense LUTs, APAE, prior and prior latents. A pre-hash checkpoint requires an
explicit verified checkpoint SHA256 exception; it does not retroactively
establish its upstream input hashes. The archived `dates_30k` and
`model_error_sf` (`s_f`) are pinned in [`assets.json`](../manifests/assets.json)
and are fetched as frozen inputs, not regenerated by the SEW stage.

The remaining scientific gates are:

1. Resolve or explicitly bound the terminal 1265 cm⁻¹ failures for all four
   affected gases: sampled maxima for CO2/H2O/N2O and exhaustive RMS for CH4.
   Full-cell scans have only been completed for CH4; the other four have
   sampled comparisons. Candidate LUTs are not yet accepted for training.
2. Independently verify the exact generator for the frozen APAE T/U/D input
   and the physical five-angle T/U/D/radiance generation upstream of the
   recovered parent. The 5.08 GB parent has a locally verified SHA256 and all
   64,350,000 selected values match the frozen radiance exactly; see the
   [full report](validation/radiance_parent_full.json). The missing 645–1250
   intermediate no longer blocks verification of this parent-to-frozen crop.
   The full noise comparison found two unequal float32 cells in 64,350,000;
   explain or accept that difference under an explicit numerical criterion
   before substituting generated noise. The direct synthesis path remains a
   candidate until its own output comparison passes.
3. Finish fresh SEW training with recorded inputs/configuration and evaluate
   that checkpoint on both fixed sets against the declared tolerances. A
   different prior must be treated explicitly as a new upstream input. Modal
   workspace `ac-wcltPGHZuWysY1tH2xbeL8` was disabled around 20:03 UTC on
   2026-09-27, blocking further execution there. The step-200 backup predates
   embedded input hashes, so any later continuation requires the documented
   explicit checkpoint-hash exception and independent upstream verification.
4. Preserve the separate historical-identity gate: the old evaluation did
   not record its checkpoint hash or complete input inventory. Passing metrics
   for the recovered checkpoint cannot establish those missing hashes.

All source code comments and docstrings in the extracted packages and
pipelines are English. The packages build independently, and local unit and
regression checks are collected under `tests/`.
