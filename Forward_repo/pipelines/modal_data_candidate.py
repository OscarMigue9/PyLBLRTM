"""Generate isolated 714--1250 cm^-1 training-data candidates on Modal.

Run ``modal run pipelines/modal_data_candidate.py --dry-run`` to inspect the
plan. A full T/U/D run requires passing five-gas LUT comparison reports. Every
output stays in the separate ``thesis-final-data-candidates`` volume; no stage
replaces a frozen asset or claims historical training-input identity.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = Path("/app")
CACHE_ROOT = Path("/cache/data")
LUT_ROOT = Path("/lut-candidates")
OUTPUT_ROOT = Path("/data-candidates")
GASES = ("H2O", "CO2", "O3", "N2O", "CH4")
COUNTS = (2, 30_000)
TUD_BATCH_SIZE = 16
NOISE_BATCH_SIZE = 256
STATUS = "candidate_historical_parity_unverified"
MAX_RMS_LOG10_SIGMA = 2.5e-5
MAX_ABS_LOG10_SIGMA = 2.5e-4

app = modal.App("thesis-final-data-candidates")
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
    .add_local_file(
        ROOT / "pipelines" / "modal_lut_compare.py",
        "/app/pipelines/modal_lut_compare.py",
        copy=True,
    )
    .add_local_file(
        ROOT / "pipelines" / "modal_data_candidate.py",
        "/app/pipelines/modal_data_candidate.py",
        copy=True,
    )
    .add_local_file(ROOT / "configs" / "final.json", "/app/configs/final.json", copy=True)
    .add_local_file(ROOT / "manifests" / "assets.json", "/app/manifests/assets.json", copy=True)
    .add_local_file(
        ROOT / "manifests" / "lut_blocks.json", "/app/manifests/lut_blocks.json", copy=True
    )
    .add_local_dir(ROOT / "frozen" / "gpu_pack", "/app/frozen/gpu_pack", copy=True)
)

data_volume = modal.Volume.from_name("wikithesis-datos", create_if_missing=True)
lut_volume = modal.Volume.from_name("thesis-final-lut-candidates")
output_volume = modal.Volume.from_name("thesis-final-data-candidates", create_if_missing=True)
rclone_secret = modal.Secret.from_name("rclone-wikithesis")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(root: Path) -> dict:
    """Hash the exact generation code and manifests uploaded to this image."""
    relative = [
        Path("pipelines/__init__.py"),
        Path("pipelines/assets.py"),
        Path("pipelines/modal_lut_full.py"),
        Path("pipelines/modal_lut_compare.py"),
        Path("pipelines/modal_data_candidate.py"),
        Path("configs/final.json"),
        Path("manifests/assets.json"),
        Path("manifests/lut_blocks.json"),
    ]
    relative.extend(
        path.relative_to(root)
        for path in sorted((root / "pipelines" / "generation").iterdir())
        if path.is_file() and path.suffix in (".py", ".cu")
    )
    files = {path.as_posix(): _sha256(root / path) for path in relative}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": dict(sorted(files.items()))}


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


def _tile_plan(count: int) -> tuple[dict, ...]:
    """Match the fixed 50 cm^-1 tiles and 2145 IASI channel indices."""
    if count not in COUNTS:
        raise ValueError(f"count must be one of {COUNTS}")
    channels = np.arange(714.0, 1250.0 + 1e-9, 0.25)
    tiles = []
    for start in range(695, 1250, 50):
        end = min(start + 50, 1250)
        mask = (channels >= max(start, 714) - 1e-9) & (
            channels <= end + 1e-9 if end == 1250 else channels < end - 1e-9
        )
        indices = np.where(mask)[0]
        if indices.size:
            tiles.append(
                {
                    "name": f"c{start}.npz",
                    "start_cm1": start,
                    "ca": int(indices[0]),
                    "cb": int(indices[-1]) + 1,
                }
            )
    if len(tiles) != 12 or tiles[0]["ca"] != 0 or tiles[-1]["cb"] != 2145:
        raise AssertionError("Unexpected IASI T/U/D tile plan")
    return tuple(tiles)


def _validate_chunk(path: Path, tile: dict, count: int) -> None:
    channels = np.arange(714.0, 1250.0 + 1e-9, 0.25)
    width = tile["cb"] - tile["ca"]
    with np.load(path, allow_pickle=False) as chunk:
        if (
            set(chunk.files) != {"ca", "cb", "nu", "D", "T", "U"}
            or int(chunk["ca"]) != tile["ca"]
            or int(chunk["cb"]) != tile["cb"]
            or not np.array_equal(chunk["nu"], channels[tile["ca"] : tile["cb"]])
            or chunk["T"].shape != (5, count, width)
            or chunk["U"].shape != (5, count, width)
            or chunk["D"].shape != (count, width)
            or any(not np.isfinite(chunk[key]).all() for key in ("T", "U", "D"))
        ):
            raise ValueError(f"Invalid candidate T/U/D chunk: {path}")


def _existing_receipt(path: Path, receipt_path: Path, expected: dict) -> dict | None:
    """Reuse only complete output sets with matching bytes and provenance."""
    if not path.is_file() or not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("output_sha256") != _sha256(path)
        or receipt.get("output_bytes") != path.stat().st_size
    ):
        raise ValueError(f"Stored candidate or provenance differs: {path}")
    return receipt


def _orphan(path: Path, root: Path) -> None:
    """Preserve an incomplete publication before a fresh candidate replaces it."""
    if not path.exists():
        return
    destination = root / "orphans"
    destination.mkdir(parents=True, exist_ok=True)
    prefix = _sha256(path)[:16]
    target = destination / f"{path.name}.{prefix}"
    suffix = 1
    while target.exists():
        target = destination / f"{path.name}.{prefix}.{suffix}"
        suffix += 1
    path.replace(target)


def _publish_one(source: Path, destination: Path, receipt_path: Path, receipt: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _orphan(destination, destination.parent)
    _orphan(receipt_path, destination.parent)
    pending = destination.with_name(destination.name + ".pending")
    shutil.copyfile(source, pending)
    if _sha256(pending) != receipt["output_sha256"]:
        raise ValueError("Candidate changed while copying to the output volume")
    pending.replace(destination)
    _write_json(receipt_path, receipt)
    output_volume.commit()


def _setup_remote() -> None:
    encoded = os.environ.get("RCLONE_CONF_B64")
    plain = os.environ.get("RCLONE_CONF")
    if not encoded and not plain:
        raise RuntimeError("Modal secret must supply RCLONE_CONF_B64 or RCLONE_CONF")
    directory = Path.home() / ".config" / "rclone"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "rclone.conf").write_bytes(
        base64.b64decode(encoded) if encoded else plain.encode("utf-8")
    )


def _verified_assets(*identifiers: str) -> dict[str, str]:
    from pipelines.assets import load_manifest, materialize, verify_file

    assets = {item.identifier: item for item in load_manifest(APP_ROOT / "manifests/assets.json")}
    hashes = {}
    for identifier in identifiers:
        artifact = assets[identifier]
        source = materialize(artifact, CACHE_ROOT, None, "auto", APP_ROOT)
        verify_file(source, artifact)
        hashes[identifier] = artifact.sha256
    data_volume.commit()
    return hashes


def _comparison_scope(report: dict, allow_terminal_outside_model_band: bool) -> str:
    """Accept a full gate, or isolate the 1265 cm^-1 terminal for any model gas."""
    comparison = report.get("comparison", {})
    gate = comparison.get("gate", {})
    if (
        gate.get("rms_log10_sigma_limit") != MAX_RMS_LOG10_SIGMA
        or gate.get("max_abs_log10_sigma_limit") != MAX_ABS_LOG10_SIGMA
    ):
        raise ValueError("LUT comparison limits differ from the frozen gate")
    blocks = comparison.get("blocks", [])
    selection = report.get("sample_selection", {})
    terminal = comparison.get("terminal_max_abs_log10_sigma")
    maximum = comparison.get("max_abs_log10_sigma")
    rms = comparison.get("rms_log10_sigma")
    if (
        len(blocks) != 63
        or [block.get("lower_cm1") for block in blocks] != list(range(635, 1265, 10))
        or any(
            block.get("passed") is not True
            or not isinstance(block.get("rms_log10_sigma"), (int, float))
            or not isinstance(block.get("max_abs_log10_sigma"), (int, float))
            or not math.isfinite(block["rms_log10_sigma"])
            or not math.isfinite(block["max_abs_log10_sigma"])
            or block["rms_log10_sigma"] > MAX_RMS_LOG10_SIGMA
            or block["max_abs_log10_sigma"] > MAX_ABS_LOG10_SIGMA
            for block in blocks
        )
        or selection.get("terminal_global_channel_index") != 630_000
        or selection.get("channel_spacing_cm1") != 0.001
        or selection.get("state_count") != 20
        or selection.get("channel_count") != 63 * 301 + 1
        or not isinstance(rms, (float, int))
        or not isinstance(terminal, (float, int))
        or not isinstance(maximum, (float, int))
        or not all(math.isfinite(value) for value in (rms, terminal, maximum))
        or rms > MAX_RMS_LOG10_SIGMA
    ):
        raise ValueError("LUT block samples or comparison metadata failed")
    if gate.get("passed") is True:
        if maximum > MAX_ABS_LOG10_SIGMA or terminal > MAX_ABS_LOG10_SIGMA:
            raise ValueError("LUT full gate flag disagrees with its maximum")
        return "full_band_sampled_gate_pass"
    if (
        not allow_terminal_outside_model_band
        or terminal <= MAX_ABS_LOG10_SIGMA
        or maximum != terminal
    ):
        raise ValueError("LUT comparison failed beyond the isolated 1265 cm^-1 terminal")
    return "all_block_samples_pass_terminal_1265_excluded_explicitly"


def _verified_lut_reports(
    reports: dict[str, str], allow_terminal_outside_model_band: bool
) -> dict[str, dict]:
    """Check all block bytes and link each passing comparison to its candidate."""
    from pipelines.assets import load_manifest
    from pipelines.modal_lut_compare import (
        generation_source_snapshot,
        planned_blocks,
        verify_candidate,
    )

    if set(reports) != set(GASES):
        raise ValueError("Five passing gas comparison reports are required")
    assets = {item.identifier: item for item in load_manifest(APP_ROOT / "manifests/assets.json")}
    builder_source = generation_source_snapshot(APP_ROOT)
    verified = {}
    for gas in GASES:
        candidate = verify_candidate(
            LUT_ROOT,
            gas,
            planned_blocks(APP_ROOT / "manifests/lut_blocks.json", gas),
            assets,
            builder_source,
        )
        comparison = json.loads(reports[gas])
        if (
            comparison.get("schema_version") != 1
            or comparison.get("status") != "candidate_full_gas_sampled_comparison"
            or comparison.get("gas") != gas
            or comparison.get("archive", {}).get("sha256") != assets[f"lut_{gas.lower()}"].sha256
            or comparison.get("archive", {}).get("manifest_sha256")
            != _sha256(APP_ROOT / "manifests/assets.json")
            or comparison.get("archive", {}).get("block_manifest_sha256")
            != _sha256(APP_ROOT / "manifests/lut_blocks.json")
            or any(
                comparison.get("candidate", {}).get(key) != candidate[key]
                for key in (
                    "report_sha256",
                    "dense_sha256",
                    "dense_bytes",
                    "block_sha256",
                    "receipt_sha256",
                    "input_sha256",
                    "source_snapshot",
                )
            )
        ):
            raise ValueError(f"{gas} LUT comparison does not match the verified candidate")
        scope = _comparison_scope(comparison, allow_terminal_outside_model_band)
        verified[gas] = {
            "comparison_report_sha256": hashlib.sha256(reports[gas].encode("utf-8")).hexdigest(),
            "candidate_report_sha256": candidate["report_sha256"],
            "candidate_dense_sha256": candidate["dense_sha256"],
            "archived_dense_sha256": assets[f"lut_{gas.lower()}"].sha256,
            "comparison_scope": scope,
            "full_band_gate_passed": comparison["comparison"]["gate"]["passed"],
            "terminal_1265_max_abs_log10_sigma": comparison["comparison"][
                "terminal_max_abs_log10_sigma"
            ],
        }
    return verified


def _cli_command(*arguments: str, report: Path) -> list[str]:
    return [
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


def _run_cli(*arguments: str, report: Path) -> dict:
    subprocess.run(_cli_command(*arguments, report=report), cwd=APP_ROOT, check=True)
    return json.loads(report.read_text(encoding="utf-8"))


def _dataset_root(count: int) -> Path:
    _tile_plan(count)
    return OUTPUT_ROOT / f"n{count}"


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65536,
    timeout=24 * 3600,
    volumes={
        str(CACHE_ROOT): data_volume,
        str(LUT_ROOT): lut_volume,
        str(OUTPUT_ROOT): output_volume,
    },
    secrets=[rclone_secret],
)
def build_tud(
    count: int, reports: dict[str, str], allow_terminal_outside_model_band: bool = False
) -> dict:
    """Build twelve restartable all-angle T/U/D tiles from verified LUT factors."""
    sys.path.insert(0, str(APP_ROOT))
    data_volume.reload()
    lut_volume.reload()
    output_volume.reload()
    _setup_remote()
    tiles = _tile_plan(count)
    snapshot = _source_snapshot(APP_ROOT)
    inputs = _verified_assets("states_30k", "r_self", "gpu_pack_continuum")
    inputs["lut"] = _verified_lut_reports(reports, allow_terminal_outside_model_band)
    root = _dataset_root(count)
    destination = root / "tud_chunks"
    destination.mkdir(parents=True, exist_ok=True)
    completed = {}
    for tile in tiles:
        path = destination / tile["name"]
        expected = {
            "schema_version": 1,
            "status": STATUS,
            "stage": "tud_chunk",
            "count": count,
            "batch_size": TUD_BATCH_SIZE,
            "tile": tile,
            "source_snapshot": snapshot,
            "input_sha256": inputs,
        }
        receipt = _existing_receipt(path, path.with_suffix(".json"), expected)
        if receipt is not None:
            _validate_chunk(path, tile, count)
            completed[tile["name"]] = receipt

    final_path = root / "tud_report.json"
    if final_path.is_file():
        final = json.loads(final_path.read_text(encoding="utf-8"))
        if (
            final.get("schema_version") != 1
            or final.get("status") != STATUS
            or final.get("stage") != "tud"
            or final.get("count") != count
            or final.get("batch_size") != TUD_BATCH_SIZE
            or final.get("source_snapshot") != snapshot
            or final.get("input_sha256") != inputs
            or final.get("chunk_sha256")
            != {name: receipt["output_sha256"] for name, receipt in completed.items()}
            or len(completed) != len(tiles)
        ):
            raise ValueError("Stored T/U/D report or provenance differs")
        return {"action": "reused", "report": final}

    with tempfile.TemporaryDirectory(prefix=f"candidate-tud-n{count}-") as work:
        work_root = Path(work)
        block_dir = work_root / "lut_blocks"
        block_dir.mkdir()
        for gas in GASES:
            for block in sorted((LUT_ROOT / gas / "blocks").glob("lut_*.npz")):
                (block_dir / block.name).symlink_to(block)
        chunk_dir = work_root / "tud_chunks"
        chunk_dir.mkdir()
        for name in completed:
            shutil.copyfile(destination / name, chunk_dir / name)
        report_path = work_root / "build_tud.json"
        command = _cli_command(
            "build-tud",
            "--blocks-dir",
            str(block_dir),
            "--output-dir",
            str(chunk_dir),
            "--count",
            str(count),
            "--batch-size",
            str(TUD_BATCH_SIZE),
            report=report_path,
        )
        process = subprocess.Popen(command, cwd=APP_ROOT)
        try:
            exit_seen = False
            while True:
                for tile in tiles:
                    name = tile["name"]
                    source = chunk_dir / name
                    if name in completed or not source.is_file():
                        continue
                    _validate_chunk(source, tile, count)
                    receipt = {
                        "schema_version": 1,
                        "status": STATUS,
                        "stage": "tud_chunk",
                        "count": count,
                        "batch_size": TUD_BATCH_SIZE,
                        "tile": tile,
                        "source_snapshot": snapshot,
                        "input_sha256": inputs,
                        "output_sha256": _sha256(source),
                        "output_bytes": source.stat().st_size,
                    }
                    target = destination / name
                    _publish_one(source, target, target.with_suffix(".json"), receipt)
                    completed[name] = receipt
                    print(
                        f"T/U/D {len(completed)}/12: {name} {receipt['output_sha256']}", flush=True
                    )
                if process.poll() is not None:
                    if exit_seen:
                        break
                    exit_seen = True
                    continue
                time.sleep(5)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
        if len(completed) != len(tiles):
            raise ValueError("T/U/D generator completed without all twelve chunks")
        builder = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            builder.get("status") != STATUS
            or builder.get("profiles") != count
            or builder.get("range_cm1") != [714.0, 1250.0]
            or len(builder.get("chunks", [])) != len(tiles)
            or builder.get("source_states_sha256") != inputs["states_30k"]
            or builder.get("source_ratios_sha256") != inputs["r_self"]
            or builder.get("source_continuum_sha256") != inputs["gpu_pack_continuum"]
        ):
            raise ValueError("T/U/D builder report differs from verified inputs")
    final = {
        "schema_version": 1,
        "status": STATUS,
        "stage": "tud",
        "count": count,
        "batch_size": TUD_BATCH_SIZE,
        "source_snapshot": snapshot,
        "input_sha256": inputs,
        "chunk_sha256": {name: completed[name]["output_sha256"] for name in sorted(completed)},
        "builder_report": builder,
    }
    _write_json(final_path, final)
    output_volume.commit()
    return {"action": "built", "report": final}


def _verified_tud(count: int, snapshot: dict) -> dict:
    root = _dataset_root(count)
    report_path = root / "tud_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != 1
        or report.get("status") != STATUS
        or report.get("stage") != "tud"
        or report.get("count") != count
        or report.get("batch_size") != TUD_BATCH_SIZE
        or report.get("source_snapshot") != snapshot
    ):
        raise ValueError("T/U/D candidate source or count differs")
    chunks = {}
    for tile in _tile_plan(count):
        path = root / "tud_chunks" / tile["name"]
        expected = {
            "schema_version": 1,
            "status": STATUS,
            "stage": "tud_chunk",
            "count": count,
            "batch_size": TUD_BATCH_SIZE,
            "tile": tile,
            "source_snapshot": snapshot,
            "input_sha256": report["input_sha256"],
        }
        receipt = _existing_receipt(path, path.with_suffix(".json"), expected)
        if receipt is None:
            raise FileNotFoundError(f"Missing complete T/U/D chunk: {path}")
        chunks[tile["name"]] = receipt["output_sha256"]
    if report.get("chunk_sha256") != chunks:
        raise ValueError("T/U/D report chunk hashes differ")
    return {"report_sha256": _sha256(report_path), "chunk_sha256": chunks}


def _existing_pair(root: Path, name: str, files: dict[str, Path], expected: dict) -> dict | None:
    receipt_path = root / f"{name}.json"
    if not receipt_path.is_file() or any(not path.is_file() for path in files.values()):
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    hashes = {key: _sha256(path) for key, path in files.items()}
    sizes = {key: path.stat().st_size for key, path in files.items()}
    if (
        any(receipt.get(key) != value for key, value in expected.items())
        or receipt.get("output_sha256") != hashes
        or receipt.get("output_bytes") != sizes
    ):
        raise ValueError(f"Stored {name} candidate or provenance differs")
    return receipt


def _publish_pair(
    root: Path, name: str, sources: dict[str, Path], targets: dict[str, Path], receipt: dict
) -> None:
    for key, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        _orphan(target, root)
        pending = target.with_name(target.name + ".pending")
        shutil.copyfile(sources[key], pending)
        if _sha256(pending) != receipt["output_sha256"][key]:
            raise ValueError(f"Candidate {key} changed during publication")
        pending.replace(target)
    receipt_path = root / f"{name}.json"
    _orphan(receipt_path, root)
    _write_json(receipt_path, receipt)
    output_volume.commit()


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=12 * 3600,
    volumes={str(CACHE_ROOT): data_volume, str(OUTPUT_ROOT): output_volume},
    secrets=[rclone_secret],
)
def assemble_candidate(count: int) -> dict:
    """Assemble selected-angle radiance and T/U/D without replacing APAE input."""
    sys.path.insert(0, str(APP_ROOT))
    data_volume.reload()
    output_volume.reload()
    _setup_remote()
    snapshot = _source_snapshot(APP_ROOT)
    tud = _verified_tud(count, snapshot)
    inputs = _verified_assets(
        "states_30k", "emissivity_iasi", "water_refractive_index", "radiances_iasiang"
    )
    inputs["tud"] = tud
    root = _dataset_root(count)
    targets = {
        "radiance": root / "radiance" / "radiances_iasiang.npz",
        "tud_selected": root / "radiance" / "tud_iasiang_selected.npy",
    }
    expected = {
        "schema_version": 1,
        "status": STATUS,
        "stage": "assemble",
        "count": count,
        "tud_view": "selected",
        "source_snapshot": snapshot,
        "input_sha256": inputs,
    }
    existing = _existing_pair(root, "assemble", targets, expected)
    if existing is not None:
        return {"action": "reused", "report": existing}
    with tempfile.TemporaryDirectory(prefix=f"candidate-assemble-n{count}-") as work:
        work_root = Path(work)
        sources = {
            "radiance": work_root / "radiances_iasiang.npz",
            "tud_selected": work_root / "tud_iasiang_selected.npy",
        }
        builder = _run_cli(
            "assemble",
            "--chunks-dir",
            str(root / "tud_chunks"),
            "--radiance-output",
            str(sources["radiance"]),
            "--tud-output",
            str(sources["tud_selected"]),
            "--count",
            str(count),
            "--tud-view",
            "selected",
            "--archive",
            str(CACHE_ROOT / "data/outputs/radiances_iasiang.npz"),
            report=work_root / "assemble.json",
        )
        hashes = {key: _sha256(path) for key, path in sources.items()}
        if (
            builder.get("status") != STATUS
            or builder.get("profiles") != count
            or builder.get("tud_view") != "selected"
            or builder.get("view_seed") != 991
            or builder.get("emissivity_seed") != 42
            or builder.get("radiance_sha256") != hashes["radiance"]
            or builder.get("tud_sha256") != hashes["tud_selected"]
            or builder.get("numerical_canary", {}).get("archived_sha256")
            != inputs["radiances_iasiang"]
        ):
            raise ValueError("Assembly report differs from verified outputs or archive")
        receipt = {
            **expected,
            "output_sha256": hashes,
            "output_bytes": {key: path.stat().st_size for key, path in sources.items()},
            "builder_report": builder,
        }
        _publish_pair(root, "assemble", sources, targets, receipt)
    return {"action": "built", "report": receipt}


@app.function(
    image=image,
    cpu=8.0,
    memory=65536,
    timeout=12 * 3600,
    volumes={str(CACHE_ROOT): data_volume, str(OUTPUT_ROOT): output_volume},
    secrets=[rclone_secret],
)
def build_noise_candidate(count: int) -> dict:
    """Rebuild fixed-seed noise and compare its arrays to the frozen archive."""
    sys.path.insert(0, str(APP_ROOT))
    data_volume.reload()
    output_volume.reload()
    _setup_remote()
    root = _dataset_root(count)
    snapshot = _source_snapshot(APP_ROOT)
    inputs = _verified_assets("noise_iasiang")
    destination = root / "noise" / "noise_iasiang.npz"
    receipt_path = root / "noise.json"
    expected = {
        "schema_version": 1,
        "status": STATUS,
        "stage": "noise",
        "count": count,
        "batch_size": NOISE_BATCH_SIZE,
        "seed": 1337,
        "source_snapshot": snapshot,
        "input_sha256": inputs,
    }
    existing = _existing_receipt(destination, receipt_path, expected)
    if existing is not None:
        return {"action": "reused", "report": existing}
    with tempfile.TemporaryDirectory(prefix=f"candidate-noise-n{count}-") as work:
        work_root = Path(work)
        source = work_root / "noise_iasiang.npz"
        builder = _run_cli(
            "build-noise",
            "--output",
            str(source),
            "--count",
            str(count),
            "--batch-size",
            str(NOISE_BATCH_SIZE),
            "--archive",
            str(CACHE_ROOT / "data/outputs/noise_iasiang.npz"),
            report=work_root / "noise.json",
        )
        if (
            builder.get("status") != STATUS
            or builder.get("profiles") != count
            or builder.get("seed") != 1337
            or builder.get("batch_size") != NOISE_BATCH_SIZE
            or builder.get("sha256") != _sha256(source)
            or builder.get("numerical_canary", {}).get("archived_sha256") != inputs["noise_iasiang"]
        ):
            raise ValueError("Noise report differs from verified output or archive")
        receipt = {
            **expected,
            "output_sha256": builder["sha256"],
            "output_bytes": source.stat().st_size,
            "builder_report": builder,
        }
        _publish_one(source, destination, receipt_path, receipt)
    return {"action": "built", "report": receipt}


def _load_comparisons(
    directory: Path, allow_terminal_outside_model_band: bool = False
) -> dict[str, str]:
    reports = {}
    for gas in GASES:
        path = directory / f"lut_full_{gas.lower()}.json"
        content = path.read_text(encoding="utf-8")
        report = json.loads(content)
        if report.get("gas") != gas:
            raise ValueError(f"Expected {gas} LUT report: {path}")
        try:
            _comparison_scope(report, allow_terminal_outside_model_band)
        except ValueError as exc:
            raise ValueError(f"Passing {gas} LUT report required: {path}: {exc}") from exc
        reports[gas] = content
    return reports


@app.local_entrypoint()
def main(
    stage: str = "plan",
    count: int = 30_000,
    validation_dir: str = "docs/validation",
    dry_run: bool = False,
    allow_terminal_outside_model_band: bool = False,
) -> None:
    """Plan or run candidate stages; 2 profiles are a separate smoke namespace."""
    if stage not in ("plan", "tud", "assemble", "noise", "all"):
        raise ValueError("stage must be plan, tud, assemble, noise or all")
    tiles = _tile_plan(count)
    directory = Path(validation_dir)
    if stage == "plan" or dry_run:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "count": count,
                    "candidate_volume": "thesis-final-data-candidates",
                    "namespace": f"n{count}",
                    "tud_tiles": [tile["name"] for tile in tiles],
                    "requires_five_passing_lut_reports": stage in ("tud", "all"),
                    "allow_terminal_outside_model_band": allow_terminal_outside_model_band,
                    "lut_report_directory": str(directory),
                    "stages": ["tud", "assemble", "noise"] if stage == "all" else [stage],
                    "status": STATUS,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if stage in ("tud", "all"):
        reports = _load_comparisons(directory, allow_terminal_outside_model_band)
        print(
            json.dumps(
                build_tud.remote(count, reports, allow_terminal_outside_model_band),
                indent=2,
                sort_keys=True,
            )
        )
    if stage in ("assemble", "all"):
        print(json.dumps(assemble_candidate.remote(count), indent=2, sort_keys=True))
    if stage in ("noise", "all"):
        print(json.dumps(build_noise_candidate.remote(count), indent=2, sort_keys=True))
