"""Build one complete five-gas-model LUT candidate on a persistent Modal volume.

Run one gas at a time from ``thesis final``::

    modal run pipelines/modal_lut_full.py --gas CO2

The 63 ten-cm^-1 blocks are independent Modal calls. Completed blocks are reused
only after their bytes, pinned inputs and uploaded source snapshot are verified.
The dense expansion is a separate CPU call. All outputs remain candidates until
the full five-gas numerical and retrieval comparisons have passed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/app")
CACHE_ROOT = Path("/cache/data")
OUTPUT_ROOT = Path("/candidates")
MODEL_GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
PACK_IDS = ("gpu_pack_lines", "gpu_pack_qtab", "gpu_pack_continuum")

app = modal.App("thesis-final-lut-candidates")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("rclone", "ca-certificates")
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
        ROOT / "pipelines" / "modal_lut_full.py", "/app/pipelines/modal_lut_full.py", copy=True
    )
    .add_local_file(ROOT / "configs" / "final.json", "/app/configs/final.json", copy=True)
    .add_local_file(ROOT / "manifests" / "assets.json", "/app/manifests/assets.json", copy=True)
    .add_local_file(
        ROOT / "manifests" / "lut_blocks.json", "/app/manifests/lut_blocks.json", copy=True
    )
    .add_local_dir(ROOT / "frozen" / "gpu_pack", "/app/frozen/gpu_pack", copy=True)
)

data_volume = modal.Volume.from_name("wikithesis-datos", create_if_missing=True)
candidate_volume = modal.Volume.from_name("thesis-final-lut-candidates", create_if_missing=True)
rclone_secret = modal.Secret.from_name("rclone-wikithesis")


def planned_blocks(manifest: Path, gas: str) -> tuple[dict, ...]:
    """Return the manifest's exact 63-block sequence for a model gas."""
    if gas not in MODEL_GASES:
        raise ValueError(f"gas must be one of {MODEL_GASES}")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    grid = document.get("grid", {})
    if (
        document.get("schema_version") != 1
        or document.get("hash_algorithm") != "sha256"
        or grid.get("start_cm1") != 635
        or grid.get("stop_cm1") != 1265
        or grid.get("block_width_cm1") != 10
        or tuple(document.get("gases", ())) != MODEL_GASES
        or document.get("total_blocks") != 315
        or len(document.get("blocks", ())) != 315
    ):
        raise ValueError("Unexpected frozen LUT block manifest")
    records = tuple(item for item in document["blocks"] if item.get("gas") == gas)
    if len(records) != 63:
        raise ValueError(f"Expected 63 {gas} blocks")
    for index, item in enumerate(records):
        start = 635 + 10 * index
        end = start + 10
        if (
            item.get("lower_cm1") != start
            or item.get("upper_cm1") != end
            or item.get("filename") != f"lut_{gas}_{start}_{end}.npz"
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] <= 0
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in item["sha256"])
        ):
            raise ValueError(f"Invalid archived {gas} block at index {index}")
    return records


def source_snapshot(root: Path) -> dict:
    """Hash exactly the code and configuration uploaded into this Modal image."""
    relative_files = [
        Path("pipelines/__init__.py"),
        Path("pipelines/assets.py"),
        Path("pipelines/modal_lut_full.py"),
        Path("configs/final.json"),
        Path("manifests/assets.json"),
        Path("manifests/lut_blocks.json"),
    ]
    relative_files.extend(
        path.relative_to(root)
        for path in sorted((root / "pipelines" / "generation").iterdir())
        if path.is_file() and path.suffix in (".py", ".cu")
    )
    files = {path.as_posix(): _sha256(root / path) for path in relative_files}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": dict(sorted(files.items()))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _setup_remote() -> None:
    encoded = os.environ.get("RCLONE_CONF_B64")
    plain = os.environ.get("RCLONE_CONF")
    if not encoded and not plain:
        raise RuntimeError("Modal secret must supply RCLONE_CONF_B64 or RCLONE_CONF")
    config_dir = Path.home() / ".config" / "rclone"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "rclone.conf").write_bytes(
        base64.b64decode(encoded) if encoded else plain.encode("utf-8")
    )


def _verified_pack() -> dict[str, str]:
    """Materialize manifest sources, then validate every byte before generation."""
    from pipelines.assets import load_manifest, materialize, verify_file

    artifacts = {
        item.identifier: item for item in load_manifest(APP_ROOT / "manifests/assets.json")
    }
    hashes = {}
    for identifier in PACK_IDS:
        artifact = artifacts[identifier]
        source = materialize(artifact, CACHE_ROOT, None, "auto", APP_ROOT)
        verify_file(source, artifact)
        hashes[identifier] = artifact.sha256
    data_volume.commit()
    return hashes


def _existing_block(
    block: Path, receipt_path: Path, gas: str, start: int, snapshot: dict, inputs: dict
) -> dict | None:
    """Reuse only a completed block with matching bytes and provenance."""
    if not block.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("status") != "candidate_historical_source_unverified"
        or receipt.get("gas") != gas
        or receipt.get("start_cm1") != start
        or receipt.get("end_cm1") != start + 10
        or receipt.get("source_snapshot") != snapshot
        or receipt.get("input_sha256") != inputs
        or receipt.get("candidate_sha256") != _sha256(block)
    ):
        raise ValueError(f"Stored candidate or provenance differs: {block}")
    return receipt


def _run_cli(*arguments: str, report: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "pipelines.generation",
        "--manifest",
        str(APP_ROOT / "manifests/assets.json"),
        "--config",
        str(APP_ROOT / "configs/final.json"),
        "--cache",
        str(CACHE_ROOT),
        "--report",
        str(report),
        *arguments,
    ]
    subprocess.run(command, cwd=APP_ROOT, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=3600,
    volumes={str(CACHE_ROOT): data_volume, str(OUTPUT_ROOT): candidate_volume},
    secrets=[rclone_secret],
)
def build_candidate_block(gas: str, start: int) -> dict:
    """Build or verify one candidate factor block and commit its receipt."""
    sys.path.insert(0, str(APP_ROOT))
    data_volume.reload()
    candidate_volume.reload()
    _setup_remote()
    record = next(
        (
            item
            for item in planned_blocks(APP_ROOT / "manifests/lut_blocks.json", gas)
            if item["lower_cm1"] == start
        ),
        None,
    )
    if record is None:
        raise ValueError(f"Unaligned or out-of-range block: {start}")
    snapshot = source_snapshot(APP_ROOT)
    inputs = _verified_pack()
    destination = OUTPUT_ROOT / gas / "blocks"
    destination.mkdir(parents=True, exist_ok=True)
    block = destination / record["filename"]
    receipt_path = block.with_suffix(".json")
    existing = _existing_block(block, receipt_path, gas, start, snapshot, inputs)
    if existing is not None:
        return {"action": "reused", "receipt": existing}

    with tempfile.TemporaryDirectory(prefix=f"lut-{gas}-{start}-") as work:
        work_root = Path(work)
        generated = _run_cli(
            "build-lut",
            "--gas",
            gas,
            "--start",
            str(start),
            "--end",
            str(start + 10),
            "--output-dir",
            str(work_root / "blocks"),
            report=work_root / "build.json",
        )
        source = work_root / "blocks" / record["filename"]
        result = generated["blocks"]
        if len(result) != 1 or result[0]["sha256"] != _sha256(source):
            raise ValueError("Generated block differs from its builder report")
        from pipelines.generation.lut import inspect_blocks

        inspection = inspect_blocks(source.parent, gas, start, start + 10)
        receipt = {
            "schema_version": 1,
            "status": "candidate_historical_source_unverified",
            "gas": gas,
            "start_cm1": start,
            "end_cm1": start + 10,
            "source_snapshot": snapshot,
            "input_sha256": inputs,
            "candidate_sha256": result[0]["sha256"],
            "candidate_bytes": source.stat().st_size,
            "archived_reference": record,
            "builder_report": generated,
            "inspection": inspection,
        }
        pending = block.with_suffix(".npz.partial")
        shutil.copyfile(source, pending)
        if _sha256(pending) != receipt["candidate_sha256"]:
            raise ValueError("Candidate changed while copying to the output volume")
        pending.replace(block)
        _write_json(receipt_path, receipt)
    candidate_volume.commit()
    return {"action": "built", "receipt": receipt}


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=3600,
    volumes={str(CACHE_ROOT): data_volume, str(OUTPUT_ROOT): candidate_volume},
    secrets=[rclone_secret],
)
def expand_candidate_gas(gas: str) -> dict:
    """Validate all 63 receipts and expand them to one dense candidate array."""
    sys.path.insert(0, str(APP_ROOT))
    data_volume.reload()
    candidate_volume.reload()
    _setup_remote()
    records = planned_blocks(APP_ROOT / "manifests/lut_blocks.json", gas)
    snapshot = source_snapshot(APP_ROOT)
    inputs = _verified_pack()
    output_dir = OUTPUT_ROOT / gas
    blocks_dir = output_dir / "blocks"
    receipts = []
    for record in records:
        block = blocks_dir / record["filename"]
        receipt = _existing_block(
            block,
            block.with_suffix(".json"),
            gas,
            record["lower_cm1"],
            snapshot,
            inputs,
        )
        if receipt is None:
            raise FileNotFoundError(f"Incomplete candidate block: {block}")
        receipts.append(receipt)

    from pipelines.assets import load_manifest

    assets = {item.identifier: item for item in load_manifest(APP_ROOT / "manifests/assets.json")}
    archived_dense = assets[f"lut_{gas.lower()}"]
    dense = output_dir / "dense" / f"lut_band645_{gas}.npy"
    report_path = output_dir / "report.json"
    if dense.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "candidate_full_gas_unvalidated"
            and report.get("gas") == gas
            and report.get("block_count") == 63
            and report.get("source_snapshot") == snapshot
            and report.get("input_sha256") == inputs
            and report.get("block_sha256")
            == {
                item["archived_reference"]["filename"]: item["candidate_sha256"]
                for item in receipts
            }
            and report.get("dense_sha256") == _sha256(dense)
            and report.get("dense_bytes") == dense.stat().st_size
        ):
            return {"action": "reused", "report": report}
        raise ValueError(f"Stored dense candidate or provenance differs: {dense}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"dense-{gas}-") as work:
        work_root = Path(work)
        expanded = _run_cli(
            "expand-lut",
            "--gas",
            gas,
            "--start",
            "635",
            "--end",
            "1265",
            "--blocks-dir",
            str(blocks_dir),
            "--output-dir",
            str(work_root / "dense"),
            report=work_root / "expand.json",
        )
        generated_dense = work_root / "dense" / dense.name
        if expanded["sha256"] != _sha256(generated_dense):
            raise ValueError("Expanded LUT differs from its builder report")
        dense.parent.mkdir(parents=True, exist_ok=True)
        pending = dense.with_suffix(".npy.partial")
        shutil.copyfile(generated_dense, pending)
        if _sha256(pending) != expanded["sha256"]:
            raise ValueError("Dense candidate changed while copying to the output volume")
        pending.replace(dense)
    if expanded["sha256"] != _sha256(dense):
        raise ValueError("Expanded LUT differs from its builder report")
    expanded["output"] = str(dense)
    report = {
        "schema_version": 1,
        "status": "candidate_full_gas_unvalidated",
        "gas": gas,
        "range_cm1": [635, 1265],
        "block_count": len(receipts),
        "source_snapshot": snapshot,
        "input_sha256": inputs,
        "block_sha256": {
            item["archived_reference"]["filename"]: item["candidate_sha256"] for item in receipts
        },
        "dense_sha256": expanded["sha256"],
        "dense_bytes": dense.stat().st_size,
        "archived_dense_sha256": archived_dense.sha256,
        "byte_identical_to_archived_dense": expanded["sha256"] == archived_dense.sha256,
        "expansion_report": expanded,
    }
    _write_json(report_path, report)
    candidate_volume.commit()
    return {"action": "built", "report": report}


@app.local_entrypoint()
def main(gas: str = "CO2") -> None:
    """Complete or resume one gas, then print its final candidate report."""
    records = planned_blocks(ROOT / "manifests/lut_blocks.json", gas)
    for index, record in enumerate(records, start=1):
        result = build_candidate_block.remote(gas, record["lower_cm1"])
        print(
            f"{gas} {index}/63 {record['lower_cm1']}-{record['upper_cm1']}: "
            f"{result['action']} {result['receipt']['candidate_sha256']}"
        )
    result = expand_candidate_gas.remote(gas)
    print(json.dumps(result, indent=2, sort_keys=True))
