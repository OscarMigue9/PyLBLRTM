"""Reproduce the frozen z24/Ncol training and simulated evaluation stages.

Inputs are resolved through the SHA256 asset manifest. Newly trained checkpoints
are written to a separate run directory and receive their own hash report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipelines.assets import DEFAULT_MANIFEST, Artifact, hash_file, load_manifest, verify_file
from pipelines.provenance import runtime_snapshot, source_snapshot


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "final.json"
MAR_MANIFEST = ROOT / "manifests" / "mar64_inputs.json"
FINAL_NAME = "unroll714_sew_z24_ncol"
MAR_GEO_SHA256 = "dd7909e261230f7fff3e4899db8e45595a9dffc339189f38ba6b43f58546d008"

APAE_INPUTS = ("states_30k", "tud_iasi_apae", "split_30k_seed42")
PRIOR_INPUTS = ("states_30k", "dates_30k", "split_30k_seed42")
RETRIEVAL_INPUTS = (
    "states_30k",
    "dates_30k",
    "radiances_iasiang",
    "noise_iasiang",
    "split_30k_seed42",
    "model_error_sf",
    "r_self",
    "lut_h2o",
    "lut_co2",
    "lut_o3",
    "lut_n2o",
    "lut_ch4",
)


def _config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1 or document.get("name") != FINAL_NAME:
        raise ValueError("Expected the frozen unroll714_sew z24/Ncol configuration")
    if document["state"]["dimension"] != 818 or document["spectral"]["channels"] != 2145:
        raise ValueError("Configuration is incompatible with the final model")
    return document


def _paths(manifest: Path, cache: Path, identifiers: tuple[str, ...]) -> dict[str, Path]:
    artifacts: dict[str, Artifact] = {item.identifier: item for item in load_manifest(manifest)}
    paths = {}
    for identifier in identifiers:
        artifact = artifacts[identifier]
        path = cache.joinpath(*artifact.destination.parts)
        verify_file(path, artifact)
        paths[identifier] = path
    return paths


def _checkpoint(override: Path | None, manifest: Path, cache: Path, identifier: str) -> Path:
    if override is not None:
        if not override.is_file():
            raise FileNotFoundError(override)
        return override
    return _paths(manifest, cache, (identifier,))[identifier]


def _state(paths: dict[str, Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from iasi_retrieval.state import build_state_matrix_full

    with np.load(paths["states_30k"]) as archive:
        states = {key: archive[key] for key in archive.files}
    return states, build_state_matrix_full(states)


def _dates(path: Path, states: dict[str, np.ndarray]) -> np.ndarray:
    with np.load(path) as archive:
        doy = np.asarray(archive["doy"])
        if "subset_id" in archive and "subset_id" in states:
            if not np.array_equal(archive["subset_id"], states["subset_id"]):
                raise ValueError("Dates and atmospheric states are not row-aligned")
    if doy.shape != (30_000,):
        raise ValueError("Expected one day-of-year value for each of 30,000 scenes")
    return doy


def _dataset(paths: dict[str, Path]) -> dict[str, Any]:
    from iasi_retrieval.data import load_final_dataset

    return load_final_dataset(
        paths["states_30k"],
        paths["radiances_iasiang"],
        paths["noise_iasiang"],
        split_path=paths["split_30k_seed42"],
    )


def _run_record(
    stage: str,
    config_path: Path,
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "configuration_sha256": hash_file(config_path),
        "source_snapshot": source_snapshot(ROOT),
        "runtime": runtime_snapshot(),
        "inputs": {
            key: {"path": str(path), "sha256": hash_file(path)} for key, path in inputs.items()
        },
        "outputs": {
            key: {"path": str(path), "sha256": hash_file(path)} for key, path in outputs.items()
        },
        "details": details,
    }


def _write_record(output_dir: Path, stage: str, record: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{stage}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )


def _train_apae(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from iasi_retrieval.training import train_apae

    paths = _paths(args.manifest, args.cache, APAE_INPUTS)
    _, state = _state(paths)
    tud = np.load(paths["tud_iasi_apae"], mmap_mode="r")
    with np.load(paths["split_30k_seed42"]) as archive:
        splits = {key: archive[key] for key in ("train", "val", "test")}
    model, checkpoint = train_apae(
        state,
        tud,
        splits=splits,
        epochs=config["apae"]["epochs"],
        beta=config["apae"]["beta"],
        seed=config["seeds"]["apae_training"],
        device=args.device,
    )
    del model
    output = args.output_dir / "apae.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return _run_record(
        "train_apae",
        args.config,
        paths,
        {"apae": output},
        {
            "best_validation_loss": checkpoint["best_val"],
            "test": checkpoint["test"],
            "ceiling": checkpoint["ceiling"],
            "epochs_run": len(checkpoint["history"]),
        },
    )


def _train_prior(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from iasi_retrieval.models import load_apae
    from iasi_retrieval.prior import geographic_features, predict_prior, train_prior

    paths = _paths(args.manifest, args.cache, PRIOR_INPUTS)
    apae_path = _checkpoint(args.apae, args.manifest, args.cache, "apae_checkpoint")
    states, state = _state(paths)
    features = geographic_features(states, _dates(paths["dates_30k"], states))
    apae, apae_checkpoint = load_apae(apae_path, args.device)
    prior_model, checkpoint = train_prior(
        apae,
        apae_checkpoint,
        state,
        features,
        epochs=config["prior"]["epochs"],
        batch_size=config["prior"]["batch_size"],
        lr=config["prior"]["learning_rate"],
        weight_decay=config["prior"]["weight_decay"],
        patience=config["prior"]["patience"],
        seed=config["seeds"]["prior_training"],
        device=args.device,
    )
    output = args.output_dir / "prior.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    with torch.no_grad():
        latents = predict_prior(prior_model, checkpoint, features, device=args.device)
    latent_path = args.output_dir / "prior_latents.npz"
    np.savez_compressed(
        latent_path, za=latents.cpu().numpy().astype(np.float32), apae=str(apae_path)
    )
    return _run_record(
        "train_prior",
        args.config,
        {**paths, "apae": apae_path},
        {"prior": output, "prior_latents": latent_path},
        {
            "best_validation_loss": checkpoint["best_val"],
            "test": checkpoint["test"],
            "epochs_run": len(checkpoint["history"]),
        },
    )


def _retrieval(
    args: argparse.Namespace, train: bool, *, include_selection: bool = True
) -> tuple[Any, dict, dict, dict]:
    from iasi_retrieval.assembly import build_training_retrieval, load_final_retrieval

    identifiers = RETRIEVAL_INPUTS + (
        () if train or not include_selection else ("frozen_simulated_rows",)
    )
    paths = _paths(args.manifest, args.cache, identifiers)
    apae = _checkpoint(args.apae, args.manifest, args.cache, "apae_checkpoint")
    prior = _checkpoint(args.prior, args.manifest, args.cache, "prior_checkpoint")
    paths.update({"apae": apae, "prior": prior})
    dataset = _dataset(paths)
    common = (
        dataset,
        paths["dates_30k"],
        apae,
        prior,
        paths["model_error_sf"],
        paths["lut_h2o"].parent,
        paths["r_self"],
    )
    if train:
        latents = _checkpoint(args.prior_latents, args.manifest, args.cache, "prior_latents")
        paths["prior_latents"] = latents
        retrieval = build_training_retrieval(
            *common, prior_latents_path=latents, seed=args.seed, device=args.device
        )
    else:
        sew = _checkpoint(args.sew, args.manifest, args.cache, "sew_checkpoint")
        paths["sew"] = sew
        retrieval = load_final_retrieval(
            dataset,
            paths["dates_30k"],
            apae,
            prior,
            sew,
            paths["model_error_sf"],
            paths["lut_h2o"].parent,
            paths["r_self"],
            device=args.device,
        )
    return retrieval, dataset, paths, common


def _sew_input_hashes(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, str]:
    """Bind the sew checkpoint to every verified source and selected config."""
    artifacts = {item.identifier: item for item in load_manifest(args.manifest)}
    aliases = {"apae": "apae_checkpoint", "prior": "prior_checkpoint"}
    overrides = {
        "apae": args.apae,
        "prior": args.prior,
        "prior_latents": args.prior_latents,
    }
    result = {"configuration": hash_file(args.config)}
    for identifier, path in sorted(paths.items()):
        artifact = artifacts.get(aliases.get(identifier, identifier))
        verified_cache_path = (
            args.cache.joinpath(*artifact.destination.parts) if artifact is not None else None
        )
        if (
            artifact is not None
            and overrides.get(identifier) is None
            and path.resolve() == verified_cache_path.resolve()
        ):
            # _paths and _checkpoint already compared these bytes to the manifest.
            result[identifier] = artifact.sha256
        else:
            result[identifier] = hash_file(path)
    return result


def _train_unrolled(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    from iasi_retrieval.unrolled import train_unrolled

    if args.legacy_checkpoint_sha256 is not None and args.resume_sew is None:
        raise ValueError("--legacy-checkpoint-sha256 requires --resume-sew")
    resume_sha256 = hash_file(args.resume_sew) if args.resume_sew is not None else None
    if (
        args.legacy_checkpoint_sha256 is not None
        and args.legacy_checkpoint_sha256.lower() != resume_sha256
    ):
        raise ValueError("Legacy checkpoint SHA256 does not match --resume-sew")
    retrieval, dataset, paths, _ = _retrieval(args, train=True)
    input_sha256 = _sew_input_hashes(args, paths)
    output = args.output_dir / "sew.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    history = train_unrolled(
        retrieval,
        dataset["X"],
        dataset["idx"],
        steps=config["unrolled"]["optimization_steps"],
        batch_size=config["unrolled"]["batch_size"],
        supervised_rows=config["unrolled"]["supervised_samples"],
        iterations=config["unrolled"]["training_steps"],
        seed=config["seeds"]["unrolled_training"],
        checkpoint_path=output,
        resume_path=args.resume_sew,
        input_sha256=input_sha256,
        legacy_checkpoint_sha256=args.legacy_checkpoint_sha256,
    )
    if not output.exists():
        raise RuntimeError("Unrolled trainer did not write a final checkpoint")
    return _run_record(
        "train_unrolled",
        args.config,
        paths,
        {"sew": output},
        {
            "steps": history["steps"],
            "resumed_from_step": history["resumed_from_step"],
            "resume_checkpoint_sha256": resume_sha256,
            "checkpoint_input_sha256": input_sha256,
            "legacy_resume_without_embedded_input_hashes": history[
                "legacy_resume_without_embedded_input_hashes"
            ],
            "final_supervised_loss": float(history["curve"][-1]),
            "validation_trace": history["trace"],
        },
    )


def _evaluate(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    retrieval, dataset, paths, _ = _retrieval(args, train=False)
    if args.selection == "frozen300":
        selection = json.loads(paths["frozen_simulated_rows"].read_text(encoding="utf-8"))
        rows = np.asarray(selection["filas"], dtype=np.int64)
        if rows.shape != (config["evaluation"]["simulated_frozen_selection"],):
            raise ValueError("Frozen simulated selection must contain exactly 300 rows")
        if len(np.unique(rows)) != len(rows) or not np.isin(rows, dataset["idx"]["test"]).all():
            raise ValueError("Frozen simulated selection is not a unique test subset")
    else:
        rows = dataset["idx"]["test"]
    metrics = retrieval.evaluate(
        rows, dataset["X"], iterations=args.iterations, batch_size=args.batch_size
    )
    result = _run_record(
        "evaluate_simulated",
        args.config,
        paths,
        {},
        {
            "selection": args.selection,
            "count": len(rows),
            "iterations": args.iterations,
            "metrics": metrics,
            "scientific_status": config["scientific_status"],
        },
    )
    return result


def _mar64_inputs(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Any]]:
    """Verify every sea archive before constructing the GPU forward."""
    spec = json.loads(MAR_MANIFEST.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("selection", {}).get("scene_count") != 64:
        raise ValueError("The frozen mar64 input manifest is invalid")
    scene_dir = args.mar_scene_dir or args.cache / "data" / "valstates_full"
    expected = spec["scene_files"]
    actual = sorted(scene_dir.glob("*.npz"))
    if [path.name for path in actual] != [item["name"] for item in expected]:
        raise ValueError("Sea scene filenames/order differ from mar64_inputs.json")
    inputs = {}
    for path, item in zip(actual, expected, strict=True):
        if path.stat().st_size != item["bytes"] or hash_file(path) != item["sha256"]:
            raise ValueError(f"Sea scene bytes differ from manifest: {path.name}")
        inputs[f"scene_{path.stem}"] = path
    geo = _paths(args.manifest, args.cache, ("mar64_geo",))["mar64_geo"]
    if hash_file(geo) != MAR_GEO_SHA256:
        raise ValueError("Sea geography differs from the verified Modal artifact")
    inputs["mar64_geo"] = geo
    return inputs, spec


def _mar64_semantic_digest(scenes: Any) -> str:
    records = [
        {
            "name": name,
            "lat": float(scenes.geography[index, 0]),
            "lon": float(scenes.geography[index, 1]),
            "doy": int(scenes.geography[index, 2]),
            "cloud": float(scenes.cloud[index]),
            "mu": float(scenes.mu[index]),
        }
        for index, name in enumerate(scenes.names)
    ]
    content = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _evaluate_mar64(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Run the final T4 learned retrieval on 64 hash-verified IASI sea scenes."""
    from iasi_retrieval.assembly import bind_mar64
    from iasi_retrieval.mar64 import load_mar64
    from iasi_retrieval.state import GASES, state_metrics_full

    scene_inputs, spec = _mar64_inputs(args)
    scene_dir = args.mar_scene_dir or args.cache / "data" / "valstates_full"
    scenes = load_mar64(scene_dir, scene_inputs["mar64_geo"])
    if _mar64_semantic_digest(scenes) != spec["geo"]["record_sha256"]:
        raise ValueError("Sea geography semantic digest differs from the source catalogs")
    if config["evaluation"]["real_ocean_scenes"] != len(scenes.names):
        raise ValueError("Configuration does not select exactly 64 sea scenes")
    if config["evaluation"]["selected_inference_steps"] != 4:
        raise ValueError("Final sea evaluation requires T4")

    retrieval, dataset, paths, _ = _retrieval(args, train=False, include_selection=False)
    checkpoint = torch.load(paths["sew"], map_location="cpu", weights_only=True)
    if type(checkpoint.get("paso")) is not int or checkpoint["paso"] != 1000:
        raise ValueError("Sea evaluation requires a completed 1000-step SEW checkpoint")
    if args.sew is None and (
        checkpoint.get("args", {}).get("steps") != 1000 or "metricas" not in checkpoint
    ):
        raise ValueError("The recovered historical checkpoint lacks final-step evidence")

    sea_retrieval = bind_mar64(retrieval, dataset, scenes, paths["prior"])
    rows = np.arange(len(scenes.names), dtype=np.int64)
    predictions, latents = [], []
    sea_retrieval.step.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            state, _, latent, _ = sea_retrieval.run(batch, iterations=4)
            predictions.append(state.cpu().numpy().astype(np.float64))
            latents.append(latent.cpu().numpy().astype(np.float32))
    estimate = np.concatenate(predictions)
    final_latent = np.concatenate(latents)
    metrics = state_metrics_full(estimate, scenes.truth)
    metrics["gases_dex"] = float(np.mean([metrics[f"rmse_{gas}_dex"] for gas in GASES]))
    result = {
        **metrics,
        "metodo": "nuestro",
        "prior": "mlp",
        "conjunto": "mar",
        "estado": "full",
        "k": 24,
        "iters": 4,
        "n_escenas": 64,
        "grid": "iasiang",
        "nch": 2145,
        "se_corr": "isrf",
        "filas": rows.tolist(),
        "checkpoint_paso": checkpoint["paso"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "mar64_T4.json"
    metrics_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    dump_path = args.output_dir / "mar64_T4.npz"
    np.savez_compressed(
        dump_path,
        metodo="nuestro",
        conjunto="mar",
        k=np.int64(24),
        x_hat=estimate.astype(np.float32),
        x_true=scenes.truth.astype(np.float32),
        y_obs=scenes.observed.astype(np.float32),
        P_lay=scenes.pressure_layers,
        mu=scenes.mu.astype(np.float32),
        z0=sea_retrieval.initial_latents.cpu().numpy(),
        z4=final_latent,
        filas=rows,
        nombres=np.asarray(scenes.names),
    )
    return _run_record(
        "evaluate_mar64",
        args.config,
        {**paths, **scene_inputs},
        {"metrics": metrics_path, "scene_dump": dump_path},
        {
            "count": len(rows),
            "iterations": 4,
            "metrics": metrics,
            "geo_record_sha256": _mar64_semantic_digest(scenes),
            "scientific_status": config["scientific_status"],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cache", type=Path, default=Path("data_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/final"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--apae", type=Path, help="A newly trained APAE checkpoint")
    parser.add_argument("--prior", type=Path, help="A newly trained prior checkpoint")
    parser.add_argument(
        "--prior-latents", type=Path, help="Precomputed 30,000 x 24 prior latents for training"
    )
    parser.add_argument("--sew", type=Path, help="A newly trained sew checkpoint")
    parser.add_argument("--resume-sew", type=Path, help="Resume an interrupted sew training run")
    parser.add_argument(
        "--legacy-checkpoint-sha256",
        help="Explicit SHA256 attestation for a pre-manifest sew checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    commands = parser.add_subparsers(dest="stage", required=True)
    for name in ("train-apae", "train-prior", "train-unrolled"):
        commands.add_parser(name)
    evaluation = commands.add_parser("evaluate-simulated")
    evaluation.add_argument("--selection", choices=("frozen300", "test3000"), default="frozen300")
    evaluation.add_argument("--iterations", type=int, choices=(3, 4), default=4)
    evaluation.add_argument("--batch-size", type=int, default=16)
    sea_evaluation = commands.add_parser("evaluate-mar64")
    sea_evaluation.add_argument(
        "--mar-scene-dir", type=Path, help="Directory holding the 64 verified scene archives"
    )
    sea_evaluation.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    args = parser.parse_args(argv)
    try:
        config = _config(args.config)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the selected device")
        stages = {
            "train-apae": _train_apae,
            "train-prior": _train_prior,
            "train-unrolled": _train_unrolled,
            "evaluate-simulated": _evaluate,
            "evaluate-mar64": _evaluate_mar64,
        }
        record = stages[args.stage](args, config)
        record_name = args.stage.replace("-", "_")
        if args.stage == "evaluate-simulated":
            record_name += f"_{args.selection}_T{args.iterations}"
        elif args.stage == "evaluate-mar64":
            record_name += "_T4"
        _write_record(args.output_dir, record_name, record)
        print(json.dumps(record, indent=2, sort_keys=True, default=float))
        return 0
    except (KeyError, ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
