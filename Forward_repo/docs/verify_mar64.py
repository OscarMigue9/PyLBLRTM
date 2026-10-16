"""Verify the 64 IASI sea scenes against their recorded provenance.

The checker reads source archives but never copies their contents. It can also
validate the derived geography and the historical T4 scene dump when supplied.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "mar64_inputs.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_scene_archives(scene_dir: Path, manifest: dict) -> list[dict]:
    expected = manifest["scene_files"]
    actual_names = sorted(path.name for path in scene_dir.glob("*.npz"))
    assert actual_names == [item["name"] for item in expected], "Scene names/order differ"
    assert sha256_json(expected) == manifest["scene_file_set_sha256"]

    raw_geo = []
    for item in expected:
        path = scene_dir / item["name"]
        assert path.stat().st_size == item["bytes"], f"Size differs: {path.name}"
        assert sha256_file(path) == item["sha256"], f"SHA256 differs: {path.name}"
        with np.load(path, allow_pickle=False) as scene:
            assert set(scene.files) == set(manifest["scene_key_schema"]), path.name
            for key, choices in manifest["scene_key_schema"].items():
                array = scene[key]
                observed = {
                    "shape": list(array.shape),
                    "dtype": "unicode" if array.dtype.kind == "U" else str(array.dtype),
                }
                assert observed in choices, f"Schema differs: {path.name}:{key}"
            raw_geo.append(
                {
                    "name": path.stem,
                    "lat": float(scene["lat_fov"]),
                    "lon": float(scene["lon_fov"]),
                    "cloud": float(scene["cloud_frac"]),
                    "mu": float(scene["mu"]),
                }
            )
    assert all(row["cloud"] <= manifest["loader_mode"]["max_cloud"] for row in raw_geo)
    return raw_geo


def check_geo_archive(path: Path, raw_geo: list[dict], manifest: dict) -> None:
    spec = manifest["geo"]
    with np.load(path, allow_pickle=False) as geo:
        assert set(spec["required_keys"]).issubset(geo.files)
        assert all(len(geo[key]) == spec["expected_length"] for key in spec["required_keys"])
        names = [str(value) for value in geo["nombres"]]
        assert names == [row["name"] for row in raw_geo]
        for field in ("lat", "lon", "cloud", "mu"):
            expected = np.asarray([row[field] for row in raw_geo])
            assert np.allclose(geo[field], expected, rtol=1e-8, atol=1e-8), field
        records = [
            {
                "name": name,
                "lat": float(geo["lat"][i]),
                "lon": float(geo["lon"][i]),
                "doy": int(geo["doy"][i]),
                "cloud": float(geo["cloud"][i]),
                "mu": float(geo["mu"][i]),
            }
            for i, name in enumerate(names)
        ]
    assert sha256_json(records) == spec["record_sha256"], "Geography content differs"
    if spec["file_sha256"] is not None:
        assert sha256_file(path) == spec["file_sha256"], "Geography file hash differs"


def check_catalogs(catalog_dir: Path, raw_geo: list[dict], manifest: dict) -> None:
    try:
        import netCDF4
    except ImportError as exc:
        raise SystemExit("netCDF4 is needed for --catalog-dir") from exc

    spec = manifest["geo"]
    for item in spec["catalogs"]:
        path = catalog_dir / Path(item["repository_path"]).name
        assert path.stat().st_size == item["bytes"], path.name
        assert sha256_file(path) == item["sha256"], path.name

    cache = {}
    try:
        records = []
        for row in raw_geo:
            subset, index_text = row["name"].rsplit("_", 1)
            if subset not in cache:
                cache[subset] = netCDF4.Dataset(catalog_dir / f"cams_5000_{subset}_2025.nc")
            catalog = cache[subset]
            index = int(index_text)
            day = dt.date(
                int(catalog["year"][index]),
                int(catalog["month"][index]),
                int(catalog["day"][index]),
            )
            records.append({**row, "doy": day.timetuple().tm_yday})
    finally:
        for catalog in cache.values():
            catalog.close()
    assert sha256_json(records) == spec["record_sha256"], "Catalog geography differs"


def check_historical_dump(scene_dir: Path, dump_path: Path, raw_geo: list[dict]) -> None:
    band = np.arange(714.0, 1250.0 + 0.125, 0.25)
    states = []
    radiances = []
    pressures = []
    for row in raw_geo:
        with np.load(scene_dir / f"{row['name']}.npz", allow_pickle=False) as scene:
            state = np.concatenate(
                [scene["T_levels"]]
                + [np.log10(np.clip(scene["Ncol"][i], 1.0e6, None)) for i in range(5)]
                + [np.asarray([scene["T_skin"]])]
            )
            states.append(state.astype(np.float32))
            radiances.append(np.interp(band, scene["nu"], scene["L_obs"] * 1000.0))
            pressures.append(scene["P_lay"].astype(np.float32))
    with np.load(dump_path, allow_pickle=False) as dump:
        assert np.array_equal(dump["filas"], np.arange(len(raw_geo)))
        assert np.array_equal(dump["x_true"], np.asarray(states))
        assert np.array_equal(dump["P_lay"], np.asarray(pressures))
        selected = np.asarray(radiances, dtype=np.float32)[:, dump["absol"]]
        assert np.array_equal(dump["y_obs"], selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--geo", type=Path)
    parser.add_argument("--catalog-dir", type=Path)
    parser.add_argument("--historical-dump", type=Path)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_geo = check_scene_archives(args.scene_dir, manifest)
    if args.geo:
        check_geo_archive(args.geo, raw_geo, manifest)
    if args.catalog_dir:
        check_catalogs(args.catalog_dir, raw_geo, manifest)
    if args.historical_dump:
        check_historical_dump(args.scene_dir, args.historical_dump, raw_geo)
    print(f"Verified {len(raw_geo)} scene files and requested provenance checks")


if __name__ == "__main__":
    main()
