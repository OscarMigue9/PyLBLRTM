# Extracted runtime validation

The separately installed `photonix` and `iasi_retrieval` packages ran T4 on a
Modal A100-80GB using SHA256-verified LUTs, frozen inputs and the recovered
step-1,000 SEW checkpoint (`96833ee7...83fb7ea2296f`). The simulated run
record is [`sim300_T4.json`](sim300_T4.json), and its source hashes are in
[`sim300_source_hashes.json`](sim300_source_hashes.json). The sea run record is
[`mar64_T4.json`](mar64_T4.json), with a direct metric file
[`mar64_T4_metrics.json`](mar64_T4_metrics.json) and a per-scene dump
[`mar64_T4.npz`](mar64_T4.npz).

| Set | Metric | Historical | Extracted package | Absolute difference | Tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| Simulated 300 | Mean five-gas RMSE, dex | 0.066223755812 | 0.066223756224 | 4.13e-10 | 0.001 |
| Simulated 300 | Temperature RMSE, K | 2.410006033 | 2.410006140 | 1.06e-7 | 0.05 |
| Simulated 300 | Skin temperature RMSE, K | 0.829344651 | 0.829343782 | 8.69e-7 | 0.05 |
| Sea 64 | Mean five-gas RMSE, dex | 0.072248843665 | 0.072248827781 | 1.59e-8 | 0.001 |
| Sea 64 | Temperature RMSE, K | 2.216882819 | 2.216883541 | 7.22e-7 | 0.05 |
| Sea 64 | Skin temperature RMSE, K | 1.340188711 | 1.340187639 | 1.07e-6 | 0.05 |

The numerical comparisons pass the declared tolerances. The combined
[`baseline_gate.json`](baseline_gate.json) records this result and the separate
historical identity failure caused by missing original hashes. The simulated
comparison is in [`sim300_comparison.json`](sim300_comparison.json). The sea
comparison in [`mar64_comparison.json`](mar64_comparison.json) verifies the
downloaded output hashes against its Modal run record and records all 64
scene-level differences from the bundled
[historical sea dump](../../frozen/mar_sew_n4.npz). The bundle has SHA256
`9bae8bbc7e392253f40253aa29f02a0586c3e72580c26996188a49acd8c66084`;
its original repository path remains recorded in
[`baseline_sew.json`](../../manifests/baseline_sew.json).
The 64 scene indices, true states, layer pressures, and radiances on the 2,121
historical valid channels match exactly. The retrieved states differ by RMSE
`5.7670e-6`, MAE `1.4385e-6`, and maximum absolute `9.1553e-5` across all
64 by 818 float32 values.

Regenerate the sea comparison after downloading a new Modal run record, metrics
file, and scene dump:

```powershell
python -m pipelines.mar64_compare --out docs/validation/mar64_comparison.json
```

The command checks the archived dump against its baseline SHA256, both
downloaded outputs against the run record, scene indices and order, true
states, layer pressures, and observed radiances on the archived valid channels.
It then reports per-scene state differences and the frozen metric tolerances.
The regenerated `max_scene_block_metric_change` uses the official Ncol
layout (temperature 0:137, gases 137:817, skin temperature 817). The earlier
comparison mis-indexed only those block-change values; its SHA, alignment,
overall retrieved-state difference, and gate metric values were unaffected.

The sea run used `val64_geo.npz` with verified byte SHA256
`dd7909e261230f7fff3e4899db8e45595a9dffc339189f38ba6b43f58546d008`.
The 64 scene NPZs are pinned individually in
[`mar64_inputs.json`](../../manifests/mar64_inputs.json). For a fresh Modal
volume, `evaluate-mar64` fetches any missing scene from the listed Drive
directory, checks size and SHA256, and rejects incorrect existing files before
inference. The Drive directory metadata also matches the entire 64-file
manifest; the fetch checks the downloaded bytes independently.

The current simulated and sea runs use the configuration with declared metric
tolerances (SHA256 `0ff7cdee7f0b5bbae938f8a46a1418c3c7bf976a24554e5f958313ecc507015e`).
Its direct gate input is [`sim300_T4_metrics.json`](sim300_T4_metrics.json),
adapted from the run record by the following command:

```powershell
python -m pipelines.report_adapter --run-record docs/validation/sim300_T4.json --out docs/validation/sim300_T4_metrics.json
```

The checkpoint hash of the original historical training run is unavailable.
This evidence establishes numerical agreement for the recovered checkpoint;
it does not establish byte identity with the historical training checkpoint.
Regeneration of the large LUTs and radiances from raw inputs is a separate
reproducibility check.

## Fresh SEW run status

The fresh SEW training stopped at step 200 when the `meltraslav08` workspace
(`ac-wcltPGHZuWysY1tH2xbeL8`) exceeded its spend limit. The
[interruption receipt](sew_step200_interrupted.json) records the backed-up
checkpoint and upstream hashes. The [migration receipt](sew_migration_samtr7.json)
records all four artifacts transferred to the enabled `samtr7` workspace and
downloaded back with matching byte counts and SHA256. Its CPU and A100-80GB
probes passed. The source APAE/prior runs used an NVIDIA A100 80GB PCIe;
`samtr7` reports an NVIDIA A100-SXM4-80GB. Resume app
`ap-31MuiofGKfRyiIpnmq3UHP` passed all 12 requested asset fetches by SHA256
and produced a verified step-300 checkpoint. The app was deliberately stopped
after that checkpoint. Its 28,449,780-byte local backup has SHA256
`7082cf94c3395e4e65dd83e8e8f6df6712c1b739776dcd9bcb43bc036a41fca6`,
`paso=300`, 16 embedded input hashes and scheduler epoch 300. Detached resume
app `ap-HRMLdXgqKY22UkradkUpLP` initialized from it without the legacy hash
exception. No checkpoint beyond step 300 or new-model evaluation has been
verified yet.

The [full noise reconstruction report](noise_rebuild.json) compares a fresh
seed-1337 CPU build against the SHA256-verified frozen NPZ. All 64,350,000
noise values were checked: 2 differ (RMS `1.1609829838e-13`, maximum
`9.313225746e-10`); the 2,145-channel `nu` and `sigma_ch` arrays match
exactly. The report includes both differing indices and values. Its
[`noise_compare` CLI](../../pipelines/noise_compare.py) also verifies the
candidate file against the generation report before comparison.

The original 645–2760 cm⁻¹ five-angle radiance parent was located on the UIS
Drive. The [range canary](radiance_parent_canary.json) first matched four
complete selected rows. The parent was then downloaded in full: the
5,076,600,128-byte local file independently matches SHA256
`7e33eca95e2d1116041d5361d8969e31acdef3dc49e76008b48e0e4707453dd8`.
The [full comparison](radiance_parent_full.json) checked the seed-991 angle
choice for all 30,000 profiles and every one of the 64,350,000 selected
714–1250 cm⁻¹ float32 radiance values against the frozen archive. There are
zero unequal values (RMS and maximum difference `0`). This verifies the
archived parent-to-frozen selection/crop exactly. The physical generation of
the five-angle parent and its T/U/D remains a separate open gate. The verifier
is in
[`radiance_parent_canary.py`](../../pipelines/radiance_parent_canary.py).

## Candidate five-gas LUT comparisons

The [O3](lut_full_o3.json), [CO2](lut_full_co2.json),
[H2O](lut_full_h2o.json) and [N2O](lut_full_n2o.json) reports compare sampled
cells in all 63 generated blocks and the 714–1250 cm⁻¹ model band. O3 passes
the sampled 635–1265 cm⁻¹ gate. CO2, H2O and N2O pass every block and the
model-band samples; their full-range maximum fails only at the 1265 cm⁻¹
terminal channel (`5.23e-4`, `2.90e-4`, `8.70e-4` against `2.5e-4`). These
four gases have **not** had an exhaustive full-cell scan.

Modal stopped during CH4 dense expansion, after all 63 block files and receipts
were saved. They were recovered and checked against SHA256 and the original
source snapshot. A local NumPy 2.2.5 expansion produced SHA256
`b6ccc049b20bf9f1f91dc3f327c29727540ec496d54b52d73a34a007872ac187`.
The [CH4 local report](lut_full_ch4_local.json) compares it with the
[`lut_ch4` asset](../../manifests/assets.json) from the pinned Drive path,
SHA256 `4585dca36e91fee0e124056917e7f657afc01c0f6db053bf6782254dc1b8e9fe`.
A stale file with the same name under `/outputs/` had a different hash and was
excluded. The sampled CH4 gate passes.

The CH4 report also exhaustively scans all 20 × 145 × 630,001 float32 cells.
All 63 blocks and the exact 714–1250 model band pass: band RMS
`6.856176760e-7`, maximum `7.629394531e-6`. Whole-array RMS is
`7.1508474435e-7` and maximum `1.7929077148e-4`, both within their limits.
At 1265 cm⁻¹, the RMS is `3.6505779316e-5`, above the `2.5e-5` limit;
the terminal maximum is `1.7929077148e-4`, below `2.5e-4`. The exhaustive
**full-range gate fails** for CH4 because of terminal RMS. This local report
does not establish numerical parity of newly generated radiances or authorize
replacement of the frozen training LUT.
