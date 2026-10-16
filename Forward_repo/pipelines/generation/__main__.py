"""Command line entry point for candidate historical data generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipelines.assets import Artifact, load_manifest, verify_file

from .candidate_gpu import build_block, compare_block
from .gpu_pack import build_gpu_pack
from .lut import (
    GASES,
    build_self_broadening,
    check_pilot_hashes,
    expand_lut,
    inspect_blocks,
    write_report,
)
from .raw import build_emissivity, build_states
from .synthesis import assemble_radiance, build_noise, build_tud_chunks, compare_generated_npz


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "final.json"
DEFAULT_MANIFEST = ROOT / "manifests" / "assets.json"
DEFAULT_RAW_CATALOG = ROOT / "manifests" / "raw_catalogs.json"


def _artifact_path(artifacts: dict[str, Artifact], cache: Path, identifier: str) -> Path:
    artifact = artifacts[identifier]
    path = cache.joinpath(*artifact.destination.parts)
    verify_file(path, artifact)
    return path


def _source_path(
    artifacts: dict[str, Artifact], cache: Path, identifier: str, override: Path | None = None
) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(override)
        verify_file(override, artifacts[identifier])
        return override
    return _artifact_path(artifacts, cache, identifier)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--report", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight", help="Verify frozen cached source assets")
    preflight.add_argument("ids", nargs="*", default=[])

    canary = commands.add_parser("canary", help="Check archived factor identity and coverage")
    canary.add_argument("--blocks-dir", type=Path, required=True)
    canary.add_argument("--candidate", type=Path)
    canary.add_argument("--archived", type=Path)

    rself = commands.add_parser("build-rself", help="Recreate the self-broadening ratio table")
    rself.add_argument("--output", type=Path, required=True)

    gpu_pack = commands.add_parser(
        "build-gpu-pack", help="Rebuild the byte-identical historical spectroscopy pack"
    )
    gpu_pack.add_argument("--par", type=Path)
    gpu_pack.add_argument("--coupling", type=Path)
    gpu_pack.add_argument("--output-dir", type=Path, required=True)

    build = commands.add_parser("build-lut", help="Build candidate SVD factor blocks on CUDA")
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--gas", choices=GASES, required=True)
    build.add_argument("--start", type=int, required=True)
    build.add_argument("--end", type=int, required=True)
    build.add_argument("--overwrite", action="store_true")

    expand = commands.add_parser("expand-lut", help="Expand SVD factors to a dense log10 LUT")
    expand.add_argument("--blocks-dir", type=Path, required=True)
    expand.add_argument("--output-dir", type=Path, required=True)
    expand.add_argument("--gas", choices=GASES, required=True)
    expand.add_argument("--start", type=int, default=635)
    expand.add_argument("--end", type=int, default=1265)
    expand.add_argument("--overwrite", action="store_true")

    tud = commands.add_parser("build-tud", help="Synthesize candidate all-angle T/U/D chunks")
    tud.add_argument("--blocks-dir", type=Path, required=True)
    tud.add_argument("--output-dir", type=Path, required=True)
    tud.add_argument("--states", type=Path)
    tud.add_argument("--rself", type=Path)
    tud.add_argument("--count", type=int)
    tud.add_argument("--batch-size", type=int, default=16)

    assemble = commands.add_parser("assemble", help="Assemble angle-matched radiance and T/U/D")
    assemble.add_argument("--chunks-dir", type=Path, required=True)
    assemble.add_argument("--radiance-output", type=Path, required=True)
    assemble.add_argument("--tud-output", type=Path, required=True)
    assemble.add_argument("--states", type=Path)
    assemble.add_argument("--count", type=int)
    assemble.add_argument("--tud-view", choices=("selected", "nadir"), default="selected")
    assemble.add_argument("--archive", type=Path)

    noise = commands.add_parser("build-noise", help="Build frozen-seed correlated instrument noise")
    noise.add_argument("--output", type=Path, required=True)
    noise.add_argument("--count", type=int, default=30000)
    noise.add_argument("--batch-size", type=int, default=256)
    noise.add_argument("--archive", type=Path)

    states = commands.add_parser(
        "build-states", help="Rebuild states from six verified CAMS subsets"
    )
    states.add_argument("--workspace-root", type=Path, required=True)
    states.add_argument("--raw-catalog", type=Path, default=DEFAULT_RAW_CATALOG)
    states.add_argument("--output", type=Path, required=True)
    states.add_argument("--per-subset", type=int, default=5000)
    states.add_argument("--archive", type=Path)

    emissivity = commands.add_parser("build-emissivity", help="Rebuild ordered ECOSTRESS library")
    emissivity.add_argument("--workspace-root", type=Path, required=True)
    emissivity.add_argument("--raw-catalog", type=Path, default=DEFAULT_RAW_CATALOG)
    emissivity.add_argument("--record-ids", type=Path)
    emissivity.add_argument("--output", type=Path, required=True)
    emissivity.add_argument("--count", type=int)
    emissivity.add_argument("--archive", type=Path)

    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config["schema_version"] != 1 or config["spectral"]["grid"] != "iasiang":
            raise ValueError("Expected the frozen final iasiang configuration")
        artifacts = {item.identifier: item for item in load_manifest(args.manifest)}
        cache = args.cache
        if args.command == "preflight":
            ids = args.ids or [
                "gpu_pack_lines",
                "gpu_pack_qtab",
                "gpu_pack_continuum",
                "states_30k",
                "emissivity_iasi",
                "water_refractive_index",
                "r_self",
            ]
            report = {
                identifier: str(_artifact_path(artifacts, cache, identifier)) for identifier in ids
            }
        elif args.command == "canary":
            report = check_pilot_hashes(args.blocks_dir)
            report["coverage"] = [
                inspect_blocks(args.blocks_dir, gas, 635.0, 1265.0)
                for gas in config["lut"]["gases"]
            ]
            if (args.candidate is None) != (args.archived is None):
                raise ValueError(
                    "Provide both --candidate and --archived for a numerical comparison"
                )
            if args.candidate is not None:
                report["comparison"] = compare_block(args.candidate, args.archived)
        elif args.command == "build-rself":
            source = _artifact_path(artifacts, cache, "gpu_pack_lines")
            report = build_self_broadening(source, args.output)
            report["archived_sha256"] = artifacts["r_self"].sha256
            report["byte_identical_to_archive"] = report["sha256"] == artifacts["r_self"].sha256
        elif args.command == "build-gpu-pack":
            par = _source_path(artifacts, cache, "aer_lines", args.par)
            coupling = _source_path(artifacts, cache, "aer_coupling", args.coupling)
            report = build_gpu_pack(
                par,
                coupling,
                args.output_dir,
                asset_manifest=args.manifest,
            )
        elif args.command == "build-lut":
            pack = cache / "inputs" / "gpu_pack"
            for identifier in ("gpu_pack_lines", "gpu_pack_qtab", "gpu_pack_continuum"):
                _artifact_path(artifacts, cache, identifier)
            if args.start >= args.end or (args.end - args.start) % 10:
                raise ValueError("The requested block range must be a positive multiple of 10")
            results = []
            for start in range(args.start, args.end, 10):
                output = args.output_dir / f"lut_{args.gas}_{start}_{start + 10}.npz"
                results.append(
                    build_block(pack, output, args.gas, start, start + 10, overwrite=args.overwrite)
                )
            report = {"status": "candidate_historical_source_unverified", "blocks": results}
        elif args.command == "expand-lut":
            output = args.output_dir / f"lut_band645_{args.gas}.npy"
            report = expand_lut(
                args.blocks_dir, output, args.gas, args.start, args.end, overwrite=args.overwrite
            )
            frozen_id = f"lut_{args.gas.lower()}"
            if (args.start, args.end) == (635, 1265) and frozen_id in artifacts:
                report["archived_sha256"] = artifacts[frozen_id].sha256
                report["byte_identical_to_archive"] = (
                    report["sha256"] == artifacts[frozen_id].sha256
                )
        elif args.command == "build-tud":
            states = _source_path(artifacts, cache, "states_30k", args.states)
            ratios = _source_path(artifacts, cache, "r_self", args.rself)
            continuum = _artifact_path(artifacts, cache, "gpu_pack_continuum")
            report = build_tud_chunks(
                states,
                args.blocks_dir,
                ratios,
                continuum,
                args.output_dir,
                count=args.count,
                batch_size=args.batch_size,
            )
        elif args.command == "assemble":
            states = _source_path(artifacts, cache, "states_30k", args.states)
            emissivity = _artifact_path(artifacts, cache, "emissivity_iasi")
            water = _artifact_path(artifacts, cache, "water_refractive_index")
            report = assemble_radiance(
                args.chunks_dir,
                states,
                emissivity,
                water,
                args.radiance_output,
                args.tud_output,
                view_seed=config["seeds"]["view_angle_assignment"],
                emissivity_seed=config["seeds"]["emissivity_assignment"],
                tud_view=args.tud_view,
                count=args.count,
            )
            if args.count is None and args.tud_view == "selected":
                report["archived_radiance_sha256"] = artifacts["radiances_iasiang"].sha256
                report["byte_identical_to_archive"] = (
                    report["radiance_sha256"] == artifacts["radiances_iasiang"].sha256
                )
            if args.archive is not None:
                archive = _source_path(artifacts, cache, "radiances_iasiang", args.archive)
                report["numerical_canary"] = compare_generated_npz(
                    args.radiance_output, archive, kind="radiance", profiles=report["profiles"]
                )
        elif args.command == "build-noise":
            report = build_noise(
                args.output,
                count=args.count,
                batch_size=args.batch_size,
                seed=config["seeds"]["instrument_noise"],
            )
            if args.count == 30000:
                report["archived_sha256"] = artifacts["noise_iasiang"].sha256
                report["byte_identical_to_archive"] = (
                    report["sha256"] == artifacts["noise_iasiang"].sha256
                )
            if args.archive is not None:
                archive = _source_path(artifacts, cache, "noise_iasiang", args.archive)
                report["numerical_canary"] = compare_generated_npz(
                    args.output, archive, kind="noise", profiles=report["profiles"]
                )
        elif args.command == "build-states":
            archive = (
                _source_path(artifacts, cache, "states_30k", args.archive)
                if args.archive is not None
                else None
            )
            report = build_states(
                args.workspace_root,
                args.raw_catalog,
                args.output,
                per_subset=args.per_subset,
                archived=archive,
            )
        else:
            record_ids = _source_path(artifacts, cache, "emissivity_record_ids", args.record_ids)
            archive = (
                _source_path(artifacts, cache, "emissivity_iasi", args.archive)
                if args.archive is not None
                else None
            )
            report = build_emissivity(
                args.workspace_root,
                args.raw_catalog,
                record_ids,
                args.output,
                count=args.count,
                archived=archive,
            )
        if args.report is not None:
            write_report(args.report, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (KeyError, ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
