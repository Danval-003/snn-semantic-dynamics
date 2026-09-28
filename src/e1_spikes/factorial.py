from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import EventEncoder, normalize
from .distance import multiscale_distance, rate_distance
from .e11 import bootstrap_interval, load_configs
from .model import HierarchicalSNN


CELL_NAMES = {
    ("similar", "similar"): "orth+_semantic+",
    ("similar", "different"): "orth+_semantic-",
    ("different", "similar"): "orth-_semantic+",
    ("different", "different"): "orth-_semantic-",
}


def levenshtein(left: str, right: str) -> int:
    left, right = normalize(left), normalize(right)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def orthographic_similarity(left: str, right: str) -> float:
    maximum = max(len(normalize(left)), len(normalize(right)), 1)
    return 1.0 - levenshtein(left, right) / maximum


def load_pairs(path: Path):
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            pairs.append(json.loads(line))
    return pairs


def audit(pairs):
    cells = {name: [] for name in CELL_NAMES.values()}
    seen = set()
    for pair in pairs:
        key = tuple(sorted((normalize(pair["left"]), normalize(pair["right"]))))
        if key in seen:
            raise ValueError(f"Par duplicado: {key}")
        seen.add(key)
        cell = CELL_NAMES[(pair["orthography"], pair["semantics"])]
        similarity = orthographic_similarity(pair["left"], pair["right"])
        cells[cell].append({**pair, "similarity": similarity})
    counts = {name: len(rows) for name, rows in cells.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Celdas desbalanceadas: {counts}")
    report = {"counts": counts, "cells": {}}
    for name, rows in cells.items():
        lengths = [(len(normalize(row["left"])), len(normalize(row["right"]))) for row in rows]
        similarities = [row["similarity"] for row in rows]
        report["cells"][name] = {
            "orthographic_similarity_mean": float(np.mean(similarities)),
            "orthographic_similarity_min": float(np.min(similarities)),
            "orthographic_similarity_max": float(np.max(similarities)),
            "mean_pair_length": float(np.mean([np.mean(value) for value in lengths])),
            "mean_absolute_length_difference": float(
                np.mean([abs(left - right) for left, right in lengths])
            ),
            "pos": dict(Counter(row["pos"] for row in rows)),
            "exposure": dict(Counter(row["exposure"] for row in rows)),
            "relations": dict(Counter(row["relation"] for row in rows)),
        }
    similar_values = [
        row["similarity"] for name, rows in cells.items() if name.startswith("orth+") for row in rows
    ]
    different_values = [
        row["similarity"] for name, rows in cells.items() if name.startswith("orth-") for row in rows
    ]
    report["orthography_separation"] = {
        "similar_mean": float(np.mean(similar_values)),
        "different_mean": float(np.mean(different_values)),
        "gap": float(np.mean(similar_values) - np.mean(different_values)),
    }
    return cells, report


@torch.no_grad()
def evaluate_seed(model, encoder, cells, base):
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    result = {"cells": {}}
    for cell, pairs in cells.items():
        left_events = encoder.encode([pair["left"] for pair in pairs])
        right_events = encoder.encode([pair["right"] for pair in pairs])
        left_layers = model(left_events, return_all=True)
        right_layers = model(right_events, return_all=True)
        result["cells"][cell] = []
        for layer_number, (left, right) in enumerate(zip(left_layers, right_layers, strict=True), start=1):
            temporal = multiscale_distance(left, right, taus, weights)
            rates = rate_distance(left, right)
            result["cells"][cell].append(
                {
                    "layer": layer_number,
                    "van_rossum_mean": temporal.mean().item(),
                    "rate_mean": rates.mean().item(),
                    "van_rossum_per_pair": temporal.tolist(),
                    "rate_per_pair": rates.tolist(),
                }
            )
    result["effects"] = {"van_rossum": [], "rate": []}
    for layer_index in range(3):
        for metric, field in (("van_rossum", "van_rossum_mean"), ("rate", "rate_mean")):
            a = result["cells"]["orth+_semantic+"][layer_index][field]
            b = result["cells"]["orth+_semantic-"][layer_index][field]
            c = result["cells"]["orth-_semantic+"][layer_index][field]
            d = result["cells"]["orth-_semantic-"][layer_index][field]
            semantic_effect = ((b + d) - (a + c)) / 2.0
            orthographic_effect = ((c + d) - (a + b)) / 2.0
            grand_mean = (a + b + c + d) / 4.0
            result["effects"][metric].append(
                {
                    "layer": layer_index + 1,
                    "semantic_effect": semantic_effect,
                    "orthographic_effect": orthographic_effect,
                    "abstraction_index": semantic_effect - orthographic_effect,
                    "semantic_effect_relative": semantic_effect / (grand_mean + 1e-8),
                    "orthographic_effect_relative": orthographic_effect / (grand_mean + 1e-8),
                    "abstraction_index_relative": (
                        semantic_effect - orthographic_effect
                    ) / (grand_mean + 1e-8),
                    "semantic_effect_when_orth_similar": b - a,
                    "semantic_effect_when_orth_different": d - c,
                }
            )
    return result


def aggregate(seed_results, samples):
    result = {"cells": {}, "effects": {"van_rossum": [], "rate": []}}
    for cell in CELL_NAMES.values():
        result["cells"][cell] = []
        for layer_index in range(3):
            row = {"layer": layer_index + 1}
            for field in ("van_rossum_mean", "rate_mean"):
                values = [seed["cells"][cell][layer_index][field] for seed in seed_results]
                row[field] = bootstrap_interval(values, samples)
            result["cells"][cell].append(row)
    for metric in ("van_rossum", "rate"):
        for layer_index in range(3):
            row = {"layer": layer_index + 1}
            fields = (
                "semantic_effect",
                "orthographic_effect",
                "abstraction_index",
                "semantic_effect_relative",
                "orthographic_effect_relative",
                "abstraction_index_relative",
                "semantic_effect_when_orth_similar",
                "semantic_effect_when_orth_different",
            )
            for field in fields:
                values = [seed["effects"][metric][layer_index][field] for seed in seed_results]
                row[field] = bootstrap_interval(values, samples)
            result["effects"][metric].append(row)
    return result


def run(config_path="configs/factorial.toml"):
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    pairs = load_pairs(Path(config["dataset"]))
    cells, audit_report = audit(pairs)
    encoder = EventEncoder(**base["input"])
    seed_results = []
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in e11["seeds"]:
        model = HierarchicalSNN(
            encoder.channels, **base["model"], tau_mode="heterogeneous_learnable"
        )
        model.load_state_dict(
            torch.load(
                Path(config["checkpoint_dir"]) / f"seed_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        evaluation = evaluate_seed(model, encoder, cells, base)
        seed_results.append(evaluation)
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps({"seed": seed, **evaluation}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "experiment": "E1-factorial-orthography-x-semantics-pilot",
        "status": "diagnostic pilot; not a causal factorial test",
        "limitations": [
            "semantic+ uses inflection in orth+ and synonymy/paraphrase in orth-",
            "supervision exposure differs across cells",
            "lexical frequency is not controlled",
            "only 12 curated pairs per cell",
        ],
        "seeds": e11["seeds"],
        "dataset_audit": audit_report,
        "aggregate": aggregate(seed_results, config["bootstrap_samples"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Matriz 2x2 ortografia por semantica")
    parser.add_argument("--config", default="configs/factorial.toml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
