from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a DIGIMON checkout with the paper backend manifest."
    )
    parser.add_argument("--digimon-root", required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return failure when a critical file differs.",
    )
    args = parser.parse_args()

    backend_root = Path(__file__).resolve().parent
    manifest = json.loads(
        (backend_root / "compatibility_manifest.json").read_text(encoding="utf-8-sig")
    )
    digimon_root = Path(args.digimon_root).expanduser().resolve()
    differences: list[str] = []

    for record in manifest["files"]:
        path = digimon_root / record["path"]
        if not path.exists():
            differences.append(f"missing: {record['path']}")
            continue
        actual = sha256(path)
        if actual != record["sha256"]:
            differences.append(f"changed: {record['path']}")

    if differences:
        print("\n".join(differences))
        print(
            "Compatibility note: differences may be valid, but exact historical "
            "parity is not guaranteed."
        )
        return 1 if args.strict else 0

    print("DIGIMON critical files match the historical experiment manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
