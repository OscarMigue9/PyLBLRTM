"""Modal entry point for the frozen z24/Ncol reproduction stages.

Run from the project root with ``modal run pipelines/modal_final.py``. The data
and output volumes survive between stages; the asset manifest verifies every
download before a training or evaluation command starts.
"""

from __future__ import annotations

from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = "/app"
DATA_ROOT = "/cache"
RUN_ROOT = "/runs/final"

app = modal.App("thesis-final-sew-z24")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("rclone", "ca-certificates")
    .pip_install(
        "numpy==2.1.3",
        "scipy==1.14.1",
        "numba==0.61.2",
        "torch==2.7.1",
        "cupy-cuda12x==13.3.0",
    )
    .add_local_dir(
        ROOT / "packages" / "photonix",
        "/app/packages/photonix",
        copy=True,
        ignore=["**/__pycache__/**", "**/build/**", "**/*.egg-info/**"],
    )
    .add_local_dir(
        ROOT / "packages" / "iasi_retrieval",
        "/app/packages/iasi_retrieval",
        copy=True,
        ignore=["**/__pycache__/**", "**/build/**", "**/*.egg-info/**"],
    )
    .add_local_dir(ROOT / "pipelines", "/app/pipelines", copy=True, ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "configs", "/app/configs", copy=True)
    .add_local_dir(ROOT / "manifests", "/app/manifests", copy=True)
    .add_local_dir(ROOT / "frozen", "/app/frozen", copy=True)
    .run_commands("pip install --no-deps /app/packages/photonix /app/packages/iasi_retrieval")
)

data_volume = modal.Volume.from_name("wikithesis-datos", create_if_missing=True)
run_volume = modal.Volume.from_name("wikithesis-salida", create_if_missing=True)
rclone_secret = modal.Secret.from_name("rclone-wikithesis")


def _resume_sew_path(
    stage: str,
    enabled: bool,
    run_root: str | Path,
    legacy_checkpoint_sha256: str = "",
) -> Path | None:
    """Resolve only a persisted training checkpoint from the mounted run volume."""
    if legacy_checkpoint_sha256 and not enabled:
        raise ValueError("--legacy-checkpoint-sha256 requires --resume-sew")
    if not enabled:
        return None
    if stage != "train-unrolled":
        raise ValueError("--resume-sew applies only to train-unrolled")
    path = Path(run_root) / "sew.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No persisted sew checkpoint at {path}")
    return path


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=12 * 3600,
    volumes={"/cache/data": data_volume, "/runs": run_volume},
    secrets=[rclone_secret],
)
def run_stage(
    stage: str,
    use_trained_upstream: bool = False,
    selection: str = "frozen300",
    resume_sew: bool = False,
    legacy_checkpoint_sha256: str = "",
) -> dict:
    """Fetch verified inputs and run one full training or evaluation stage."""
    import base64
    import json
    import os
    import subprocess
    import sys

    sys.path.insert(0, APP_ROOT)
    from pipelines.final import APAE_INPUTS, PRIOR_INPUTS, RETRIEVAL_INPUTS

    allowed = (
        "train-apae",
        "train-prior",
        "train-unrolled",
        "evaluate-simulated",
        "evaluate-mar64",
    )
    if stage not in allowed:
        raise ValueError(f"stage must be one of {allowed}")
    if selection not in ("frozen300", "test3000"):
        raise ValueError("selection must be frozen300 or test3000")
    resume_path = _resume_sew_path(stage, resume_sew, RUN_ROOT, legacy_checkpoint_sha256)
    encoded = os.environ.get("RCLONE_CONF_B64")
    plain = os.environ.get("RCLONE_CONF")
    if not encoded and not plain:
        raise RuntimeError("Modal secret must supply RCLONE_CONF_B64 or RCLONE_CONF")
    config_dir = Path.home() / ".config" / "rclone"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "rclone.conf").write_bytes(
        base64.b64decode(encoded) if encoded else plain.encode("utf-8")
    )

    os.chdir(APP_ROOT)
    if stage == "train-apae":
        identifiers = list(APAE_INPUTS)
    elif stage == "train-prior":
        identifiers = list(PRIOR_INPUTS)
        if not use_trained_upstream:
            identifiers.append("apae_checkpoint")
    else:
        identifiers = list(RETRIEVAL_INPUTS)
        if stage == "train-unrolled" and not use_trained_upstream:
            identifiers.append("prior_latents")
        if stage == "evaluate-simulated":
            identifiers.append("frozen_simulated_rows")
        if stage == "evaluate-mar64":
            identifiers.append("mar64_geo")
        if not use_trained_upstream:
            identifiers.extend(("apae_checkpoint", "prior_checkpoint"))
            if stage in ("evaluate-simulated", "evaluate-mar64"):
                identifiers.append("sew_checkpoint")
    for identifier in identifiers:
        subprocess.run(
            [sys.executable, "-m", "pipelines.assets", "--cache", DATA_ROOT, "fetch", identifier],
            check=True,
        )
    if stage == "evaluate-mar64":
        from pipelines.mar64_fetch import fetch_mar64_scenes

        scenes = fetch_mar64_scenes(Path(DATA_ROOT) / "data" / "valstates_full")
        print(f"PASS mar64: {len(scenes)} verified scenes")
        data_volume.commit()
    command = [
        sys.executable,
        "-m",
        "pipelines.final",
        "--cache",
        DATA_ROOT,
        "--output-dir",
        RUN_ROOT,
        "--device",
        "cuda",
    ]
    if use_trained_upstream and stage != "train-apae":
        command.extend(("--apae", f"{RUN_ROOT}/apae.pt"))
        if stage in ("train-unrolled", "evaluate-simulated", "evaluate-mar64"):
            command.extend(("--prior", f"{RUN_ROOT}/prior.pt"))
        if stage == "train-unrolled":
            command.extend(("--prior-latents", f"{RUN_ROOT}/prior_latents.npz"))
        if stage in ("evaluate-simulated", "evaluate-mar64"):
            command.extend(("--sew", f"{RUN_ROOT}/sew.pt"))
    if resume_path is not None:
        command.extend(("--resume-sew", str(resume_path)))
        if legacy_checkpoint_sha256:
            command.extend(("--legacy-checkpoint-sha256", legacy_checkpoint_sha256))
    command.append(stage)
    if stage == "evaluate-simulated":
        command.extend(("--selection", selection))
    if stage == "evaluate-mar64":
        command.extend(("--mar-scene-dir", f"{DATA_ROOT}/data/valstates_full"))
    subprocess.run(command, check=True)
    data_volume.commit()
    run_volume.commit()
    report_name = stage.replace("-", "_")
    if stage == "evaluate-simulated":
        report_name += f"_{selection}_T4"
    elif stage == "evaluate-mar64":
        report_name += "_T4"
    report = Path(RUN_ROOT) / f"{report_name}.json"
    return json.loads(report.read_text(encoding="utf-8"))


@app.local_entrypoint()
def main(
    stage: str = "evaluate-simulated",
    use_trained_upstream: bool = False,
    selection: str = "frozen300",
    resume_sew: bool = False,
    legacy_checkpoint_sha256: str = "",
) -> None:
    """Run one selected stage in Modal and display its recorded result."""
    import json

    print(
        json.dumps(
            run_stage.remote(
                stage, use_trained_upstream, selection, resume_sew, legacy_checkpoint_sha256
            ),
            indent=2,
        )
    )
