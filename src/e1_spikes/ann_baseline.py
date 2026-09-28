from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .data import EventEncoder, batch_triplets, make_e11_training_triplets, make_fixed_validation
from .distance import multiscale_distance
from .e11 import bootstrap_interval, load_configs, split_spikes
from .e12a import distance_statistics
from .factorial import audit, evaluate_seed as evaluate_factorial, load_pairs
from .input_corruption import permute_graphemes
from .model_ann import HierarchicalANN
from .train import seed_everything


def encode_triplets(model, encoder, triplets):
    texts = []
    for item in triplets:
        texts.extend([item.anchor, item.positive, item.negative])
    activity = model(encoder.encode(texts))
    return split_spikes(activity)


def regularizer(activity, target):
    magnitude = activity.abs()
    return (magnitude.mean() - target).square() + 0.03 * magnitude.mean(dim=(0, 1)).var()


def train_one(seed, base, encoder, training, epochs):
    seed_everything(seed)
    model = HierarchicalANN(encoder.channels, **base["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=base["run"]["learning_rate"])
    rng = random.Random(seed + 1)
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    final_loss = float("nan")
    for _ in range(epochs):
        losses = []
        model.train()
        for batch in batch_triplets(training, base["run"]["batch_size"], rng):
            a, p, n = encode_triplets(model, encoder, batch)
            d_ap = multiscale_distance(a, p, taus, weights)
            d_an = multiscale_distance(a, n, taus, weights)
            semantic = functional.relu(d_ap - d_an + base["run"]["margin"]).mean()
            joined = torch.cat([a, p, n])
            loss = semantic + base["regularization"]["rate_weight"] * regularizer(
                joined, base["regularization"]["target_rate"]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        final_loss = float(np.mean(losses))
    return model, final_loss


@torch.no_grad()
def evaluate(model, encoder, validation, base, permutation_trials, seed):
    texts = []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
    events = encoder.encode(texts)
    layers = model(events, return_all=True)
    kinds = [item.kind for item in validation]
    masks = {
        "all": torch.ones(len(validation), dtype=torch.bool),
        "random_negative": torch.tensor([kind == "random_negative" for kind in kinds]),
        "hard_orthographic": torch.tensor([kind == "hard_orthographic" for kind in kinds]),
    }
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    result = {"subsets": {}, "activity": []}
    for subset, mask in masks.items():
        result["subsets"][subset] = []
        for layer_number, layer in enumerate(layers, start=1):
            a, p, n = split_spikes(layer)
            result["subsets"][subset].append(
                {
                    "layer": layer_number,
                    "temporal": distance_statistics(a[mask], p[mask], n[mask], taus, weights),
                    "integrated": distance_statistics(a[mask], p[mask], n[mask], rate_only=True),
                }
            )
    stimulus_end = encoder.max_chars * encoder.char_steps
    for layer_number, layer in enumerate(layers, start=1):
        result["activity"].append(
            {
                "layer": layer_number,
                "mean_absolute_activity": layer.abs().mean().item(),
                "active_fraction_1e-3": (layer.abs() > 1e-3).float().mean().item(),
                "post_absolute_activity": layer[:, stimulus_end:].abs().mean().item(),
            }
        )
    permutation_stas = []
    for trial in range(permutation_trials):
        generator = torch.Generator().manual_seed(seed * 100 + trial)
        corrupted = permute_graphemes(events, stimulus_end, generator)
        output = model(corrupted)
        a, p, n = split_spikes(output)
        permutation_stas.append(distance_statistics(a, p, n, rate_only=True)["sta"])
    result["permuted_integrated_sta"] = float(np.mean(permutation_stas))
    return result


def aggregate(seed_results, samples):
    summary = {"subsets": {}, "activity": []}
    for subset in ("all", "random_negative", "hard_orthographic"):
        summary["subsets"][subset] = []
        for layer_index in range(3):
            row = {"layer": layer_index + 1}
            for metric in ("temporal", "integrated"):
                row[metric] = {}
                for field in ("sta", "positive_distance", "negative_distance", "margin"):
                    values = [r["evaluation"]["subsets"][subset][layer_index][metric][field] for r in seed_results]
                    row[metric][field] = bootstrap_interval(values, samples)
            summary["subsets"][subset].append(row)
    for layer_index in range(3):
        row = {"layer": layer_index + 1}
        for field in ("mean_absolute_activity", "active_fraction_1e-3", "post_absolute_activity"):
            row[field] = bootstrap_interval(
                [r["evaluation"]["activity"][layer_index][field] for r in seed_results], samples
            )
        summary["activity"].append(row)
    summary["permuted_integrated_sta"] = bootstrap_interval(
        [r["evaluation"]["permuted_integrated_sta"] for r in seed_results], samples
    )
    return summary


def run(config_path="configs/ann.toml", epochs_override=None, seeds_override=None):
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    seeds = seeds_override or e11["seeds"]
    epochs = epochs_override or base["run"]["epochs"]
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(e11["validation_random_size"])
    factorial_cells, factorial_audit = audit(load_pairs(Path(config["factorial_dataset"])))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for seed in seeds:
        training = make_e11_training_triplets(seed, validation, e11["training_repeats"])
        model, loss = train_one(seed, base, encoder, training, epochs)
        evaluation = evaluate(
            model, encoder, validation, base, config["permutation_trials"], seed
        )
        factorial = evaluate_factorial(model, encoder, factorial_cells, base)
        result = {
            "seed": seed,
            "loss": loss,
            "parameters": sum(p.numel() for p in model.parameters()),
            "evaluation": evaluation,
            "factorial": factorial,
        }
        results.append(result)
        torch.save(model.state_dict(), output_dir / f"seed_{seed}.pt")
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"seed={seed} integrated={evaluation['subsets']['all'][2]['integrated']['sta']:.3f} "
            f"hard={evaluation['subsets']['hard_orthographic'][2]['integrated']['sta']:.3f}"
        )
    factorial_results = [r["factorial"] for r in results]
    from .factorial import aggregate as aggregate_factorial

    summary = {
        "experiment": "E1-paired-continuous-ANN-baseline",
        "unit": "adaptive leaky tanh recurrent unit",
        "seeds": seeds,
        "epochs": epochs,
        "parameters": results[0]["parameters"],
        "elapsed_seconds": time.perf_counter() - started,
        "iid": aggregate(results, config["bootstrap_samples"]),
        "factorial_audit": factorial_audit,
        "factorial": aggregate_factorial(factorial_results, config["bootstrap_samples"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Baseline ANN recurrente continuo pareado")
    parser.add_argument("--config", default="configs/ann.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.config, args.epochs, args.seeds)


if __name__ == "__main__":
    main()

