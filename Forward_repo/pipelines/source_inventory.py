"""Record hashes of only the raw CAMS and ECOSTRESS inputs used by the final model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assets import hash_file


FINAL_CAMS_SUBSETS = ("temp", "hum", "co2", "oz", "n2o", "ch4")


def _record(path: Path, root: Path) -> dict[str, str | int]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def inventory_sources(cams: Path, ecostress: Path, ids_path: Path) -> dict:
    """Hash six selected CAMS NetCDFs and the ordered ECOSTRESS records."""
    cams_files = [cams / f"cams_5000_{subset}_2025.nc" for subset in FINAL_CAMS_SUBSETS]
    missing_cams = [path for path in cams_files if not path.is_file()]
    if missing_cams:
        raise FileNotFoundError(f"Missing selected CAMS files: {missing_cams}")
    with ids_path.open("rb") as stream:
        record_ids = np.load(stream, allow_pickle=False)
    if len(record_ids) != 1419 or len(set(record_ids.tolist())) != 1419:
        raise ValueError("Expected 1419 unique ECOSTRESS record IDs")
    spectra = [ecostress / f"{record_id}.spectrum.txt" for record_id in record_ids]
    missing = [str(path) for path in spectra if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing selected ECOSTRESS spectra: {missing[:5]}")
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "status": "local_bytes_verified_source_publication_pending",
        "cams_l137": {
            "workspace_directory": "data/atmospheres/nwpsaf_cams137_2025",
            "files": [_record(path, cams) for path in cams_files],
        },
        "ecostress_selected": {
            "workspace_directory": "data/surface/ecospeclib/ecospeclib-all",
            "ordered_record_ids_sha256": hash_file(ids_path),
            "files": [_record(path, ecostress) for path in spectra],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cams", type=Path, required=True)
    parser.add_argument("--ecostress", type=Path, required=True)
    parser.add_argument("--record-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = inventory_sources(args.cams, args.ecostress, args.record_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Wrote {len(document['cams_l137']['files'])} CAMS and "
        f"{len(document['ecostress_selected']['files'])} ECOSTRESS hashes"
    )


if __name__ == "__main__":
    main()
