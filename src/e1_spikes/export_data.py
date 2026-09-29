from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .data import Triplet, make_fixed_validation
from .generalization import (
    build_context_data,
    build_relation_data,
    matched_no_context_training,
)
from .lexical_holdout import build_lexical_holdout


DEFAULT_SEED = 20260928


def serialize_triplets(items: list[Triplet]) -> str:
    return "".join(
        json.dumps({"index": index, **asdict(item)}, ensure_ascii=False, sort_keys=True) + "\n"
        for index, item in enumerate(items)
    )


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def rendered_files(seed: int = DEFAULT_SEED) -> dict[str, str]:
    relation_train, relation_test = build_relation_data(seed)
    context_train, context_test = build_context_data(seed)
    no_context = matched_no_context_training(
        relation_train, len(relation_train) + len(context_train)
    )
    datasets = {
        "fixed_validation.jsonl": make_fixed_validation(),
        "relation_train.jsonl": relation_train,
        "relation_disjoint_test.jsonl": relation_test,
        "context_exposure_train.jsonl": context_train,
        "context_transfer_test.jsonl": context_test,
        "context_no_exposure_control.jsonl": no_context,
    }
    rendered = {name: serialize_triplets(items) for name, items in datasets.items()}
    lexical_rows = build_lexical_holdout()
    rendered["lexical_holdout_pairs.jsonl"] = "".join(
        json.dumps({"index": index, **row}, ensure_ascii=False, sort_keys=True) + "\n"
        for index, row in enumerate(lexical_rows)
    )
    counts = {name: len(items) for name, items in datasets.items()}
    counts["lexical_holdout_pairs.jsonl"] = len(lexical_rows)
    manifest = {
        "schema_version": 1,
        "generator": "uv run e1-export-data",
        "seed": seed,
        "description": (
            "Deterministic exports of fixed validation, generalization triplets, "
            "and the independent lexical holdout constructed by the experiment code."
        ),
        "files": {
            name: {
                "examples": counts[name],
                "sha256": sha256(content),
            }
            for name, content in rendered.items()
        },
    }
    rendered["manifest.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    return rendered


def export(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, str]:
    files = rendered_files(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output_dir / name).write_text(content, encoding="utf-8")
    return files


def check(output_dir: Path, seed: int = DEFAULT_SEED) -> list[str]:
    expected = rendered_files(seed)
    mismatches = []
    for name, content in expected.items():
        path = output_dir / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            mismatches.append(name)
    return mismatches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export deterministic, human-auditable E1 dataset manifests"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when committed exports differ from the experiment constructors.",
    )
    args = parser.parse_args()
    if args.check:
        mismatches = check(args.output_dir, args.seed)
        if mismatches:
            raise SystemExit("Outdated or missing exports: " + ", ".join(mismatches))
        print(f"Dataset exports are current: {args.output_dir}")
        return
    files = export(args.output_dir, args.seed)
    print(f"Exported {len(files) - 1} datasets and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
