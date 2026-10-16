"""Inventory the historical compressed cross-section LUT without loading its arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import hash_file


FINAL_GASES = ("H2O", "CO2", "O3", "N2O", "CH4")


def inventory_blocks(
    directory: Path,
    gases: tuple[str, ...] = FINAL_GASES,
    start_cm1: int = 635,
    stop_cm1: int = 1265,
    width_cm1: int = 10,
) -> list[dict[str, str | int]]:
    """Hash exactly one SVD block for every gas and spectral interval."""
    if stop_cm1 <= start_cm1 or (stop_cm1 - start_cm1) % width_cm1:
        raise ValueError("The spectral range must contain complete fixed-width blocks")
    records = []
    for gas in gases:
        for lower in range(start_cm1, stop_cm1, width_cm1):
            name = f"lut_{gas}_{lower}_{lower + width_cm1}.npz"
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            records.append(
                {
                    "gas": gas,
                    "lower_cm1": lower,
                    "upper_cm1": lower + width_cm1,
                    "filename": name,
                    "bytes": path.stat().st_size,
                    "sha256": hash_file(path),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = inventory_blocks(args.directory)
    document = {
        "schema_version": 1,
        "scientific_status": "historical_source_revision_unverified",
        "hash_algorithm": "sha256",
        "grid": {"start_cm1": 635, "stop_cm1": 1265, "block_width_cm1": 10},
        "gases": list(FINAL_GASES),
        "total_blocks": len(records),
        "blocks": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} block hashes to {args.output}")


if __name__ == "__main__":
    main()
