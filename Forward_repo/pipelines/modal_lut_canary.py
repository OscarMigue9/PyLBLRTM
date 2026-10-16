"""Run one isolated 895-905 cm^-1 LUT provenance canary on Modal.

From ``thesis final``, after materializing the three frozen GPU pack assets:

    modal run pipelines/modal_lut_canary.py --gas CO2 --archived-block <verified-block.npz>

Only the three pack files and one 3 MB archived block are sent to Modal. The
candidate block returns to a new local output directory with a JSON report.
This pilot cannot establish parity for the full five-gas LUT.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
START = 895
END = 905
PACK_ASSETS = {
    "lineas.npz": "gpu_pack_lines",
    "qtab.npz": "gpu_pack_qtab",
    "cont.npz": "gpu_pack_continuum",
}
MAX_RMS_LOG10_SIGMA = 2.5e-5
MAX_ABS_LOG10_SIGMA = 2.5e-4

app = modal.App("thesis-lut-pilot-co2-895-905")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("numpy==2.1.3", "scipy==1.14.1", "cupy-cuda12x==13.3.0")
    .add_local_dir(
        ROOT / "pipelines" / "generation",
        "/app/pipelines/generation",
        copy=True,
        ignore=["**/__pycache__/**"],
    )
    .add_local_file(ROOT / "pipelines" / "__init__.py", "/app/pipelines/__init__.py", copy=True)
    .add_local_file(ROOT / "pipelines" / "assets.py", "/app/pipelines/assets.py", copy=True)
    .add_local_file(
        ROOT / "pipelines" / "modal_lut_canary.py", "/app/pipelines/modal_lut_canary.py", copy=True
    )
    .add_local_file(ROOT / "manifests" / "assets.json", "/app/manifests/assets.json", copy=True)
    .add_local_file(
        ROOT / "manifests" / "lut_blocks.json", "/app/manifests/lut_blocks.json", copy=True
    )
)


def _block_name(gas: str) -> str:
    if gas not in GASES:
        raise ValueError(f"Gas must be one of {GASES}")
    return f"lut_{gas}_{START}_{END}.npz"


def _archived_record(manifest: Path, gas: str) -> dict:
    name = _block_name(gas)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("hash_algorithm") != "sha256":
        raise ValueError("Unsupported LUT block manifest")
    records = [item for item in document["blocks"] if item["filename"] == name]
    if len(records) != 1 or records[0]["gas"] != gas:
        raise ValueError(f"Expected one frozen {name} record")
    return records[0]


def _verify_bytes(data: bytes, size: int, digest: str, label: str) -> str:
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != size or actual != digest:
        raise ValueError(f"{label} differs from the frozen manifest")
    return actual


def assess_comparison(comparison: dict, gas: str = "CO2") -> dict:
    """Apply thresholds fixed before seeing the generated pilot block."""
    _block_name(gas)
    rms = float(comparison["rms_log10_sigma"])
    maximum = float(comparison["max_abs_log10_sigma"])
    passed = (
        math.isfinite(rms)
        and math.isfinite(maximum)
        and rms <= MAX_RMS_LOG10_SIGMA
        and maximum <= MAX_ABS_LOG10_SIGMA
    )
    return {
        "passed": passed,
        "rms_log10_sigma_limit": MAX_RMS_LOG10_SIGMA,
        "max_abs_log10_sigma_limit": MAX_ABS_LOG10_SIGMA,
        "comparison_grid": "20 deterministic state indices x 301 deterministic channel indices",
        "scope": f"one {gas} {START}-{END} cm^-1 candidate block only",
    }


@app.function(image=image, gpu="A100-80GB", cpu=8.0, memory=65536, timeout=3600)
def run_pilot(pack_bytes: dict[str, bytes], archive_bytes: bytes, gas: str) -> tuple[dict, bytes]:
    """Build and compare one block without persistent Modal volumes."""
    import sys

    sys.path.insert(0, "/app")
    from pipelines.assets import load_manifest, verify_file
    from pipelines.generation.candidate_gpu import build_block, compare_block

    frozen = {item.identifier: item for item in load_manifest(Path("/app/manifests/assets.json"))}
    block_name = _block_name(gas)
    archived = _archived_record(Path("/app/manifests/lut_blocks.json"), gas)
    input_hashes = {}
    with tempfile.TemporaryDirectory(prefix="lut-pilot-") as work:
        root = Path(work)
        pack = root / "pack"
        pack.mkdir()
        if set(pack_bytes) != set(PACK_ASSETS):
            raise ValueError("Expected exactly three GPU pack files")
        for name, identifier in PACK_ASSETS.items():
            artifact = frozen[identifier]
            input_hashes[name] = _verify_bytes(
                pack_bytes[name], artifact.size, artifact.sha256, name
            )
            path = pack / name
            path.write_bytes(pack_bytes[name])
            verify_file(path, artifact)
        input_hashes[block_name] = _verify_bytes(
            archive_bytes, int(archived["bytes"]), archived["sha256"], block_name
        )
        reference = root / "archived" / block_name
        reference.parent.mkdir()
        reference.write_bytes(archive_bytes)
        candidate = root / "candidate" / block_name
        result = build_block(pack, candidate, gas, START, END)
        comparison = compare_block(candidate, reference)
        output_bytes = candidate.read_bytes()
        _verify_bytes(output_bytes, len(output_bytes), result["sha256"], "candidate block")
        report = {
            "status": "candidate_pilot_comparison_only",
            "gas": gas,
            "inputs_sha256": input_hashes,
            "candidate_sha256": result["sha256"],
            "archive_sha256": archived["sha256"],
            "generator_sha256": hashlib.sha256(
                Path("/app/pipelines/generation/candidate_gpu.py").read_bytes()
            ).hexdigest(),
            "kernel_sha256": hashlib.sha256(
                Path("/app/pipelines/generation/historical_kernel.cu").read_bytes()
            ).hexdigest(),
            "build": result,
            "comparison": comparison,
            "gate": assess_comparison(comparison, gas),
        }
    return report, output_bytes


def _new_local_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@app.local_entrypoint()
def main(
    archived_block: str,
    gas: str = "CO2",
    pack_dir: str = "data_cache/inputs/gpu_pack",
    output_dir: str | None = None,
) -> None:
    """Verify local inputs, run the pilot, and save its isolated report."""
    from pipelines.assets import hash_file, load_manifest, verify_file

    pack = Path(pack_dir)
    archive = Path(archived_block)
    block_name = _block_name(gas)
    output = Path(output_dir or f"runs/lut_pilot_{gas.lower()}_{START}_{END}")
    output_block = output / block_name
    output_report = output / "report.json"
    if output_block.exists() or output_report.exists():
        raise FileExistsError(f"Pilot output already exists in {output}")
    artifacts = {item.identifier: item for item in load_manifest(ROOT / "manifests/assets.json")}
    payload = {}
    for name, identifier in PACK_ASSETS.items():
        source = pack / name
        verify_file(source, artifacts[identifier])
        payload[name] = source.read_bytes()
    reference = _archived_record(ROOT / "manifests/lut_blocks.json", gas)
    if not archive.is_file() or archive.stat().st_size != reference["bytes"]:
        raise ValueError(f"Archived block size differs: {archive}")
    if hash_file(archive) != reference["sha256"]:
        raise ValueError(f"Archived block SHA256 differs: {archive}")
    report, candidate = run_pilot.remote(payload, archive.read_bytes(), gas)
    _verify_bytes(candidate, len(candidate), report["candidate_sha256"], "returned block")
    _new_local_file(output_block, candidate)
    _new_local_file(output_report, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["gate"]["passed"]:
        raise RuntimeError("The candidate pilot exceeds the predeclared numerical limits")
