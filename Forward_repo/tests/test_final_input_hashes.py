"""The training pipeline binds a sew checkpoint to all resolved inputs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import PurePosixPath

from iasi_retrieval.unrolled import SEW_INPUT_IDENTIFIERS
from pipelines import final
from pipelines.assets import Artifact, hash_file


def test_sew_input_hashes_include_verified_assets_and_trained_overrides(tmp_path, monkeypatch):
    manifest = []
    paths = {}
    for identifier in final.RETRIEVAL_INPUTS:
        path = tmp_path / f"{identifier}.bin"
        path.write_bytes(identifier.encode("ascii"))
        paths[identifier] = path
        manifest.append(
            Artifact(
                identifier=identifier,
                destination=PurePosixPath(path.name),
                size=path.stat().st_size,
                sha256=hash_file(path),
                remote_uri=None,
                workspace_path=None,
                bundle_path=None,
                role="test",
            )
        )
    overrides = {}
    for identifier in ("apae", "prior", "prior_latents"):
        path = tmp_path / f"trained_{identifier}.bin"
        path.write_bytes(f"trained {identifier}".encode("ascii"))
        paths[identifier] = path
        overrides[identifier] = path
    config = tmp_path / "final.json"
    config.write_text("{}", encoding="utf-8")
    args = Namespace(
        manifest=tmp_path / "assets.json",
        cache=tmp_path,
        config=config,
        **overrides,
    )
    monkeypatch.setattr(final, "load_manifest", lambda _: manifest)

    hashes = final._sew_input_hashes(args, paths)

    assert set(hashes) == SEW_INPUT_IDENTIFIERS
    assert hashes["configuration"] == hash_file(config)
    for identifier, path in paths.items():
        assert hashes[identifier] == hash_file(path)
