from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import torch

from .data import EventEncoder, make_fixed_validation
from .distance import multiscale_distance, rate_distance
from .e11 import bootstrap_interval, load_configs, split_spikes
from .model import HierarchicalSNN


def load_e12a_config(path: Path):
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    return config, e11, base


def distance_statistics(a, p, n, taus=None, weights=None, rate_only=False):
    if rate_only:
        positive = rate_distance(a, p)
        negative = rate_distance(a, n)
    else:
        positive = multiscale_distance(a, p, taus, weights)
        negative = multiscale_distance(a, n, taus, weights)
    margin = negative - positive
    # El margen relativo facilita leer capas con distinta densidad de spikes,
    # sin reemplazar el margen bruto solicitado.
    relative = margin / (positive.abs() + 1e-8)
    return {
        "sta": (positive < negative).float().mean().item(),
        "positive_distance": positive.mean().item(),
        "negative_distance": negative.mean().item(),
        "margin": margin.mean().item(),
        "relative_margin": relative.mean().item(),
    }


def evaluate_subset(a, p, n, mask, variants, all_taus, all_weights):
    a, p, n = a[mask], p[mask], n[mask]
    result = {}
    for name, indices in variants.items():
        taus = [all_taus[index] for index in indices]
        weights = [all_weights[index] for index in indices]
        result[name] = {
            "taus": taus,
            "weights": weights,
            **distance_statistics(a, p, n, taus, weights),
        }
    result["rate_only"] = distance_statistics(a, p, n, rate_only=True)
    return result


@torch.no_grad()
def evaluate_seed(model, encoder, validation, variants, base):
    texts = []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
    layers = model(encoder.encode(texts), return_all=True)
    kinds = [item.kind for item in validation]
    masks = {
        "all": torch.ones(len(validation), dtype=torch.bool),
        "random_negative": torch.tensor([kind == "random_negative" for kind in kinds]),
        "hard_orthographic": torch.tensor([kind == "hard_orthographic" for kind in kinds]),
    }
    all_taus = base["distance"]["taus"]
    all_weights = base["distance"]["weights"]
    final_a, final_p, final_n = split_spikes(layers[-1])
    sensitivity = {
        subset: evaluate_subset(
            final_a, final_p, final_n, mask, variants, all_taus, all_weights
        )
        for subset, mask in masks.items()
    }

    hierarchy = {}
    for subset, mask in masks.items():
        hierarchy[subset] = []
        for layer_number, layer in enumerate(layers, start=1):
            a, p, n = split_spikes(layer)
            stats = distance_statistics(a[mask], p[mask], n[mask], all_taus, all_weights)
            hierarchy[subset].append({"layer": layer_number, **stats})
    return {"metric_sensitivity": sensitivity, "hierarchy": hierarchy}


def aggregate_values(values, samples):
    return bootstrap_interval(values, samples)


def aggregate(seed_results, variants, samples):
    sensitivity = {}
    for subset in ("all", "random_negative", "hard_orthographic"):
        sensitivity[subset] = {}
        for variant in (*variants.keys(), "rate_only"):
            sensitivity[subset][variant] = {}
            for field in ("sta", "positive_distance", "negative_distance", "margin"):
                values = [
                    result["metric_sensitivity"][subset][variant][field]
                    for result in seed_results
                ]
                sensitivity[subset][variant][field] = aggregate_values(values, samples)

    hierarchy = {}
    for subset in ("all", "random_negative", "hard_orthographic"):
        hierarchy[subset] = []
        for layer_index in range(3):
            row = {"layer": layer_index + 1}
            for field in (
                "sta",
                "positive_distance",
                "negative_distance",
                "margin",
                "relative_margin",
            ):
                values = [
                    result["hierarchy"][subset][layer_index][field]
                    for result in seed_results
                ]
                row[field] = aggregate_values(values, samples)
            hierarchy[subset].append(row)
    comparisons = {}
    pairs = (
        ("tau_14", "tau_2_6_14"),
        ("tau_14", "tau_2"),
        ("tau_14", "rate_only"),
    )
    for subset in ("all", "random_negative", "hard_orthographic"):
        comparisons[subset] = {}
        for left, right in pairs:
            values = [
                result["metric_sensitivity"][subset][left]["sta"]
                - result["metric_sensitivity"][subset][right]["sta"]
                for result in seed_results
            ]
            comparisons[subset][f"{left}_minus_{right}"] = aggregate_values(values, samples)
    return {
        "metric_sensitivity": sensitivity,
        "paired_comparisons": comparisons,
        "hierarchy": hierarchy,
    }


def rank_variants(aggregate_result):
    ranking = {}
    for subset, variants in aggregate_result["metric_sensitivity"].items():
        ranking[subset] = sorted(
            (
                {"variant": name, "sta_mean": values["sta"]["mean"]}
                for name, values in variants.items()
            ),
            key=lambda row: row["sta_mean"],
            reverse=True,
        )
    return ranking


def run(config_path: str | Path = "configs/e12a.toml"):
    config, e11, base = load_e12a_config(Path(config_path))
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(e11["validation_random_size"])
    variants = config["distance_variants"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for seed in e11["seeds"]:
        model = HierarchicalSNN(encoder.channels, **base["model"])
        model.load_state_dict(
            torch.load(checkpoint_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        )
        model.eval()
        evaluation = evaluate_seed(model, encoder, validation, variants, base)
        result = {"seed": seed, **evaluation}
        seed_results.append(result)
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        best = max(
            evaluation["metric_sensitivity"]["all"].items(),
            key=lambda item: item[1]["sta"],
        )
        print(f"seed={seed} best={best[0]} STA={best[1]['sta']:.3f}")

    aggregated = aggregate(seed_results, variants, config["bootstrap_samples"])
    summary = {
        "experiment": "E1.2A-metric-sensitivity-and-H2",
        "training": "none; evaluates frozen E1.1 checkpoints",
        "seeds": e11["seeds"],
        "validation_examples": len(validation),
        "weight_policy": "original weights restricted to included taus, then normalized",
        "aggregate": aggregated,
        "ranking": rank_variants(aggregated),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="E1.2A y margenes jerarquicos H2")
    parser.add_argument("--config", default="configs/e12a.toml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
