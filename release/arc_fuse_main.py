from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _run_with_runtime_config(release_root: Path, remaining: list[str]) -> None:
    help_only = "-h" in remaining or "--help" in remaining
    api_key = os.environ.get("ARC_FUSE_API_KEY", "").strip()
    if not api_key and not help_only:
        raise SystemExit("Set ARC_FUSE_API_KEY before running the real backend.")
    if not api_key:
        api_key = "sk-help-only"

    llm_config = {
        "api_type": "openai",
        "api_key": api_key,
        "base_url": os.environ.get(
            "ARC_FUSE_BASE_URL",
            "https://api.openai.com/v1",
        ),
        "model": os.environ.get("ARC_FUSE_MODEL", "gpt-4o-mini"),
    }

    old_home = os.environ.get("HOME")
    old_profile = os.environ.get("USERPROFILE")
    with tempfile.TemporaryDirectory(prefix="arc_fuse_config_") as temp_home:
        option_dir = Path(temp_home) / "Option"
        option_dir.mkdir(parents=True)
        config_path = option_dir / "Config2.yaml"
        config_path.write_text(
            json.dumps({"llm": llm_config}),
            encoding="utf-8",
        )
        try:
            config_path.chmod(0o600)
        except OSError:
            pass

        os.environ["HOME"] = temp_home
        os.environ["USERPROFILE"] = temp_home
        try:
            from arc_fuse_digimon.runner import main_cli

            main_cli(remaining)
        finally:
            if old_home is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old_home
            if old_profile is None:
                os.environ.pop("USERPROFILE", None)
            else:
                os.environ["USERPROFILE"] = old_profile


def bootstrap() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--digimon-root",
        default=os.environ.get("DIGIMON_ROOT", ""),
        help="Path to a DIGIMON/GraphRAG checkout.",
    )
    known, remaining = parser.parse_known_args()

    root_value = known.digimon_root.strip()
    if not root_value:
        raise SystemExit(
            "Set DIGIMON_ROOT or pass --digimon-root with a compatible "
            "DIGIMON/GraphRAG checkout."
        )

    digimon_root = Path(root_value).expanduser().resolve()
    required = [
        digimon_root / "Core" / "GraphRAG.py",
        digimon_root / "Option" / "Config2.py",
        digimon_root / "Data" / "QueryDataset.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "The supplied DIGIMON_ROOT is incompatible; missing: "
            + ", ".join(missing)
        )

    release_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(release_root))
    sys.path.insert(0, str(release_root / "research_backend"))
    sys.path.insert(0, str(digimon_root))
    os.environ["METAGPT_PROJECT_ROOT"] = str(digimon_root)
    os.chdir(digimon_root)

    _run_with_runtime_config(release_root, remaining)


if __name__ == "__main__":
    bootstrap()
