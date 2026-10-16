# Training provenance for the selected z24/Ncol model

The selected model is the dense 818-state APAE, its geographic latent prior,
and the shared `unroll714_sew` update. The frozen artifacts and their SHA256
values are listed in [`assets.json`](../manifests/assets.json). A historical
checkpoint can establish its saved metadata and weights; it cannot establish
an invocation field that was never saved.

## APAE

The frozen `apae_fix_lat24b1e-4.pt` stores its original `args`. Its values are
`state=full`, `no_tud=False`, `use_psurf=False`, `latent_dim=24`, `epochs=300`,
`batch_size=512`, `lr=0.001`, `weight_decay=0.0001`, `beta=0.0001`, `seed=7`,
and `patience=40`. It has an 818-component state and a 6,435-component
transmission/upwelling/downwelling input. The archived training history has
300 epochs, with its best validation loss `0.036749335159858065` at epoch
298. The checkpoint stores that same `best_val`.

The invocation is also present in the original
`tanda_apae_fix.sh` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/tanda_apae_fix.sh`).
The extracted [`training.py`](../packages/iasi_retrieval/src/iasi_retrieval/training.py)
uses the same AdamW defaults, stochastic APAE, seven-block state loss,
DataLoader shuffling, validation threshold (`1e-6`) and early stopping. The
fixed seed-42 row split is bundled in `frozen/`.

## Geographic prior

The frozen `prior_z_deep.pt` contains only `state_dict`, `capas`, `mu_p`,
`sd_p`, and `epocas`. Its architecture is `4-256-256-256-256-24` and
`epocas=400`. The four saved input means and standard deviations agree with
`[lat, lon, day_of_year, P_surf_hPa]` computed over the bundled training rows
(means exactly; standard deviations within `1e-12`). This establishes the
input order, train split and feature normalization.

The original prior trainer was never committed. The current
`prior_z.py` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/prior_z.py`)
was reconstructed from the checkpoint and launcher notes on 2026-08-22. Its
defaults are AdamW, `lr=0.001`, `weight_decay=0.0001`, batch 512, at most
400 epochs, patience 40, and seed 7. **The frozen prior does not record its
historical seed, optimizer, stopping epoch, or complete command.** Its first
layer does, however, retain a strong initialization fingerprint: the trained
historical `0.weight` correlates `0.947823` with a fresh network initialized
with PyTorch seed 0; `0.bias` correlates `0.9867`. Seeds 1–9, 42, and 1337
give first-layer weight correlations between `-0.0752` and `+0.0489`, while
the newly trained seed-7 network correlates `0.968458` with a fresh seed-7
initialization. This is strong
evidence that the historical network used **seed 0 immediately before its
own initialization**. It is an inference from weights, not saved metadata.
The model Git history first adds `prior_z.py` in commit `683c71e` dated
2026-08-24, after the deep-prior sweep. The Drive backups under
`WikiThesis_714/salida` and `WikiThesis_respaldo/salida`, and the inspected
Modal output volume, have the two deep-prior artifacts but no original
training script or history. Their later `source/prior_z.py` snapshots are
copies of the reconstruction. Consequently, `epocas=400` may be either a
completed count or a requested budget; the saved five-field checkpoint does
not distinguish them.
The reconstructed script seeds before loading the APAE, whereas the
historical fingerprint matches construction without those intervening random
draws. The extracted pipeline loads APAE first and then
[`train_prior`](../packages/iasi_retrieval/src/iasi_retrieval/prior.py) seeds
PyTorch, CUDA and NumPy immediately before `make_prior_network`; seed 0 in
that trainer reproduces the inferred initialization order. A local CPU
diagnostic with seed 0 stopped after 229 epochs, selected a validation loss
of `0.13689873`, and produced latents `0.34078` RMSE from the archived latents.
Its test temperature RMSE was `3.15826 K`. The trained first-layer weights
correlate `0.98925` with the archived first layer, but the output remains
different. This diagnostic does not identify the missing historical objective,
optimizer trajectory or stopping rule, so the final reproducible configuration
retains the reconstructed seed-7 recipe rather than treating seed 0 alone as
an accredited replacement.
The reconstructed script's defaults are therefore a reproducible candidate recipe, not
verified historical provenance. The original
`cadena_apae645_unrolled.sh` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/cadena_apae645_unrolled.sh`)
explicitly reports that the reconstructed prior was numerically different
from the published one. The extracted
[`prior.py`](../packages/iasi_retrieval/src/iasi_retrieval/prior.py) follows
that reconstruction. New runs record their seed and training history; the
remaining parameters are fixed by the config and module defaults.

The historical launcher notes describe prior training on a laptop CPU;
the exact runtime for the deep 256-by-4 checkpoint was not logged. The
initial retraining report ran on a Modal A100 with seed 7 and selected epoch
242 after stopping at epoch 282. Its seven-block state losses on train,
validation and test are `0.10464830`, `0.13522399`, `0.13370083`, compared
with `0.08789194`, `0.13934624`, `0.13430787` for the frozen prior when
both are evaluated through the same frozen APAE. The stronger historical
train fit and higher validation loss are consistent with longer training
without early stopping, but do not establish the lost objective or schedule.
The test temperature RMSEs are `3.13532 K` (new seed-7) and `3.08866 K`
(historical). See [`train_prior.json`](validation/train_prior.json) for the
new run and its source/input hashes. The retrained APAE used by that run has
all 30 model-state tensors bit-identical to the frozen APAE, despite a
different checkpoint file hash.

## Shared SEW update

The original step-400 `unroll714_sew_ckpt.pt` is the most reliable saved
training invocation. Its `args` record `iters=3`, `steps=1000`, `bs=16`,
`nsup=24000`, `labels=perfect`, `lam_chi2=0`, `angulo=True`,
`se_corr=isrf`, `ventanas=todas`, `sd_piso=0`, `clamp_z=6`, `lr=0.001`,
`seed=0`, `ckpt_cada=100`, `n_eval=3000`, `traza_cada=50`, and
`traza_n=256`. These agree with the original
`cadena_sew714.sh` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/cadena_sew714.sh`),
the fixed [`final.json`](../configs/final.json), and the extracted
[`unrolled.py`](../packages/iasi_retrieval/src/iasi_retrieval/unrolled.py)
defaults. Inference uses four shared updates; training used three.

The recovered step-1000 checkpoint was saved during a later `--solo-eval`
run. Its `args` therefore describe that evaluation (`iters=10`,
`traza_cada=0`, and a resume path), not the training invocation. Its weights
and optimizer state remain those of the completed training run.

There is an observable historical restart detail. The step-400 checkpoint
declares `paso=400`, but all eight AdamW parameter states have step 401. The
step-1000 checkpoint declares `paso=1000`, but its AdamW states have step
1002. The original
`exp18_unrolled.py` (historical path: `model/physinv/closure_forward/experimentos/apae_loss_fisica/exp18_unrolled.py`)
saved periodic checkpoints after an update while recording the zero-based
loop index as `paso`; resuming from it could repeat an update. The full
restart sequence is not archived. The extracted trainer records completed
updates correctly and makes 1,000 updates in a fresh run. Exact historical
weights are therefore an unsuitable retraining criterion. The accepted gate
uses the same verified inputs and configuration, then compares evaluation
metrics against the fixed tolerances in `final.json`.

The config currently records the key scientific choices, while some
implementation parameters above are fixed by the training module defaults.
The run report includes a source snapshot to make those defaults auditable.

## Step-200 checkpoint migration and continuation

The fresh SEW run stopped after 200 completed updates when the `meltraslav08`
Modal workspace (`ac-wcltPGHZuWysY1tH2xbeL8`) exceeded its spend limit. Its
checkpoint predates embedded input hashes. The
[interruption receipt](validation/sew_step200_interrupted.json) records its
SHA256, recipe fields and three saved upstream hashes. The
[migration receipt](validation/sew_migration_samtr7.json) records the SEW,
APAE, prior and prior-latent files copied to `samtr7` under
`wikithesis-salida/final`, downloaded back and checked for identical bytes
by SHA256.

The initial APAE and prior run reports identify an NVIDIA A100 80GB PCIe.
The enabled `samtr7` GPU probe identifies an NVIDIA A100-SXM4-80GB. Both
match the runner's requested A100-80GB memory class, but the hardware subtype
changed. Resume app `ap-31MuiofGKfRyiIpnmq3UHP` passed all 12 requested
asset fetches by SHA256 and produced a verified step-300 checkpoint, then was
deliberately stopped. The local backup `runs/sew_step300_backup.pt` has SHA256
`7082cf94c3395e4e65dd83e8e8f6df6712c1b739776dcd9bcb43bc036a41fca6`.
Its `paso=300`, 16 embedded input hashes and scheduler epoch 300 were checked.
Detached app `ap-HRMLdXgqKY22UkradkUpLP` initialized from that checkpoint
without the legacy hash exception. A checkpoint beyond step 300 and the two
new-model evaluations have not yet been verified. The final acceptance
criterion remains the fixed simulated-300 and sea-64 metric tolerances, not
a bitwise match of the continuation trajectory.
