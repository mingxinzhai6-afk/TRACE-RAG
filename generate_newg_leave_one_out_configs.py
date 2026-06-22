from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_BASE = Path("Option/Method/NewG.yaml")
DEFAULT_OUT_DIR = Path("Option/Method/generated_leave_one_out")


VARIANT_ORDER = [
    "full",
    "no_router",
    "no_regen",
    "no_critic",
    "no_commendor",
    "no_normalizer",
    "no_disambiguation",
    "single_agent",
]

VARIANT_LABELS = {
    "full": "Full NewG",
    "no_router": "w/o Router",
    "no_regen": "w/o Re-Generator",
    "no_critic": "w/o Critic",
    "no_commendor": "w/o Commendor",
    "no_normalizer": "w/o Answer Normalizer",
    "no_disambiguation": "w/o Entity Disambiguation",
    "single_agent": "Single judge/voter",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    if "newg" not in data or not isinstance(data["newg"], dict):
        raise ValueError(f"{path} has no top-level newg mapping")
    return data


def apply_variant(base: dict[str, Any], variant: str) -> dict[str, Any]:
    data = copy.deepcopy(base)
    cfg = data["newg"]

    if variant == "full":
        pass
    elif variant == "no_router":
        cfg["use_routing"] = False
    elif variant == "no_regen":
        cfg["use_regen"] = False
    elif variant == "no_critic":
        cfg["use_critic"] = False
        # Critic is the driver for iterative retrieval. Without it, repeated
        # rounds have no structured feedback and mostly repeat the same route.
        cfg["max_rounds"] = 1
    elif variant == "no_commendor":
        cfg["use_commendor"] = False
    elif variant == "no_normalizer":
        cfg["use_normalizer"] = False
        cfg["critic_use_normalized_answer"] = False
    elif variant == "no_disambiguation":
        cfg["use_disambiguation"] = False
    elif variant == "single_agent":
        cfg["n_judges"] = 1
        cfg["n_voters"] = 1
    else:
        raise ValueError(f"Unknown leave-one-out variant: {variant}")

    return data


def write_variant(base: dict[str, Any], out_dir: Path, variant: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"NewG_loo_{variant}.yaml"
    data = apply_variant(base, variant)
    header = (
        "################################################################################\n"
        f"# Auto-generated leave-one-out NewG config: {VARIANT_LABELS[variant]}\n"
        "# Source of truth: Option/Method/NewG.yaml\n"
        "# Regenerate with: python generate_newg_leave_one_out_configs.py\n"
        "################################################################################\n\n"
    )
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    out.write_text(header + body, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Full-minus-one NewG ablation configs from NewG.yaml."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=VARIANT_ORDER,
        choices=VARIANT_ORDER,
        help="Leave-one-out variants to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_yaml(args.base)
    for variant in args.variants:
        out = write_variant(base, args.out_dir, variant)
        print(f"{variant}\t{out}")


if __name__ == "__main__":
    main()
