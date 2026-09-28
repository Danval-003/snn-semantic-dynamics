from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import torch

from .data import EventEncoder, make_fixed_validation
from .e11 import bootstrap_interval, load_configs, split_spikes
from .e12a import distance_statistics
from .input_corruption import (
    assert_preserves_channel_counts,
    bag_of_graphemes,
    ordered_timing_jitter,
    permute_graphemes,
    reverse_graphemes,
)
from .model import HierarchicalSNN


def load_config(path: Path):
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    return config, e11, base


def make_masks(validation):
    kinds = [item.kind for item in validation]
    return {
        "all": torch.ones(len(validation), dtype=torch.bool),
        "random_negative": torch.tensor([kind == "random_negative" for kind in kinds]),
        "hard_orthographic": torch.tensor([kind == "hard_orthographic" for kind in kinds]),
    }


@torch.no_grad()
def evaluate_layers(model, events, masks, taus, weights):
    layers = model(events, return_all=True)
    result = {}
    for subset, mask in masks.items():
        result[subset] = []
        for layer_number, layer in enumerate(layers, start=1):
            a, p, n = split_spikes(layer)
            temporal = distance_statistics(a[mask], p[mask], n[mask], taus, weights)
            rates = distance_statistics(a[mask], p[mask], n[mask], rate_only=True)
            result[subset].append(
                {
                    "layer": layer_number,
                    "van_rossum": temporal,
                    "rate_only": rates,
                }
            )
    return result


def average_trials(trials):
    averaged = {}
    for subset in trials[0]:
        averaged[subset] = []
        for layer_index in range(len(trials[0][subset])):
            row = {"layer": layer_index + 1}
            for metric in ("van_rossum", "rate_only"):
                row[metric] = {}
                for field in ("sta", "positive_distance", "negative_distance", "margin"):
                    row[metric][field] = float(
                        np.mean([trial[subset][layer_index][metric][field] for trial in trials])
                    )
            averaged[subset].append(row)
    return averaged


@torch.no_grad()
def evaluate_seed(model, encoder, validation, config, base, seed):
    texts = []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
    original_events = encoder.encode(texts)
    stimulus_steps = encoder.max_chars * encoder.char_steps
    masks = make_masks(validation)
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    result = {
        "correct": evaluate_layers(model, original_events, masks, taus, weights),
        "reverse": evaluate_layers(
            model,
            reverse_graphemes(original_events, stimulus_steps),
            masks,
            taus,
            weights,
        ),
    }
    random_conditions = {
        "permuted_order": lambda events, steps, generator: permute_graphemes(
            events, steps, generator
        ),
        "ordered_jitter_mild": lambda events, steps, generator: ordered_timing_jitter(
            events, steps, generator, max_gap=3
        ),
        "ordered_jitter_strong": lambda events, steps, generator: ordered_timing_jitter(
            events, steps, generator, max_gap=5
        ),
        "bag_graphemes": lambda events, steps, generator: bag_of_graphemes(
            events, steps, generator
        ),
    }
    for condition_index, (name, corruption) in enumerate(random_conditions.items(), start=1):
        trials = []
        for trial in range(config["corruption_trials"]):
            generator = torch.Generator().manual_seed(seed * 10000 + condition_index * 100 + trial)
            corrupted = corruption(original_events, stimulus_steps, generator)
            assert_preserves_channel_counts(original_events, corrupted)
            trials.append(evaluate_layers(model, corrupted, masks, taus, weights))
        result[name] = average_trials(trials)
    return result


def aggregate(seed_results, samples):
    conditions = (
        "correct",
        "reverse",
        "permuted_order",
        "ordered_jitter_mild",
        "ordered_jitter_strong",
        "bag_graphemes",
    )
    subsets = ("all", "random_negative", "hard_orthographic")
    result = {"conditions": {}, "paired_drops": {}}
    for condition in conditions:
        result["conditions"][condition] = {}
        for subset in subsets:
            result["conditions"][condition][subset] = []
            for layer_index in range(3):
                row = {"layer": layer_index + 1}
                for metric in ("van_rossum", "rate_only"):
                    row[metric] = {}
                    for field in ("sta", "positive_distance", "negative_distance", "margin"):
                        values = [
                            seed_result[condition][subset][layer_index][metric][field]
                            for seed_result in seed_results
                        ]
                        row[metric][field] = bootstrap_interval(values, samples)
                result["conditions"][condition][subset].append(row)

    # Diferencias pareadas de la representacion conceptual final (capa 3).
    for subset in subsets:
        result["paired_drops"][subset] = {}
        for condition in conditions[1:]:
            result["paired_drops"][subset][f"correct_minus_{condition}"] = {}
            for metric in ("van_rossum", "rate_only"):
                values = [
                    seed_result["correct"][subset][2][metric]["sta"]
                    - seed_result[condition][subset][2][metric]["sta"]
                    for seed_result in seed_results
                ]
                result["paired_drops"][subset][f"correct_minus_{condition}"][metric] = (
                    bootstrap_interval(values, samples)
                )
    return result


def run(config_path: str | Path = "configs/e12c.toml"):
    config, e11, base = load_config(Path(config_path))
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(e11["validation_random_size"])
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
        evaluation = evaluate_seed(model, encoder, validation, config, base, seed)
        seed_results.append(evaluation)
        payload = {"seed": seed, "evaluation": evaluation}
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"seed={seed} correct={evaluation['correct']['all'][2]['rate_only']['sta']:.3f} "
            f"reverse={evaluation['reverse']['all'][2]['rate_only']['sta']:.3f} "
            f"perm={evaluation['permuted_order']['all'][2]['rate_only']['sta']:.3f} "
            f"jitter3={evaluation['ordered_jitter_mild']['all'][2]['rate_only']['sta']:.3f} "
            f"jitter5={evaluation['ordered_jitter_strong']['all'][2]['rate_only']['sta']:.3f} "
            f"bag={evaluation['bag_graphemes']['all'][2]['rate_only']['sta']:.3f}"
        )
    summary = {
        "experiment": "E1.2C-input-temporal-destruction",
        "training": "none; evaluates frozen E1.1 checkpoints",
        "seeds": e11["seeds"],
        "validation_examples": len(validation),
        "corruption_trials": config["corruption_trials"],
        "controls": {
            "reverse": "reverse grapheme order, preserve regular occupied times",
            "permuted_order": "permute graphemes, preserve occupied times",
            "ordered_jitter_mild": "preserve order/start, draw gaps uniformly from 1..3",
            "ordered_jitter_strong": "preserve order/start, draw gaps uniformly from 1..5",
            "bag_graphemes": "randomize both grapheme order and unique event times",
        },
        "inference_limit": (
            "Order corruption tests dependence on grapheme sequence. It does not by itself "
            "demonstrate precise output spike timing or an advantage specific to SNNs."
        ),
        "aggregate": aggregate(seed_results, config["bootstrap_samples"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="E1.2C: destruccion temporal de entrada")
    parser.add_argument("--config", default="configs/e12c.toml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
