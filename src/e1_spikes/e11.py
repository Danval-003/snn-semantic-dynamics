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

from .corruption import (
    assert_preserves_per_neuron_counts,
    global_time_shuffle,
    local_time_shuffle,
)
from .data import (
    EventEncoder,
    batch_triplets,
    make_e11_training_triplets,
    make_fixed_validation,
)
from .distance import multiscale_distance
from .metrics import triplet_metrics
from .model import HierarchicalSNN
from .train import activity_regularizer, encode_triplet_batch, seed_everything


def load_configs(path: Path):
    with path.open("rb") as handle:
        suite = tomllib.load(handle)
    base_path = Path(suite["base_config"])
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    with base_path.open("rb") as handle:
        base = tomllib.load(handle)
    return suite, base


def train_one(seed, base, encoder, train_triplets, epochs, tau_mode=None):
    seed_everything(seed)
    device = torch.device("cpu")
    model = HierarchicalSNN(encoder.channels, **base["model"], tau_mode=tau_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base["run"]["learning_rate"])
    rng = random.Random(seed + 1)
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        losses = []
        for batch in batch_triplets(train_triplets, base["run"]["batch_size"], rng):
            a, p, n = encode_triplet_batch(model, encoder, batch, device)
            d_ap = multiscale_distance(a, p, taus, weights)
            d_an = multiscale_distance(a, n, taus, weights)
            semantic = functional.relu(d_ap - d_an + base["run"]["margin"]).mean()
            joined = torch.cat([a, p, n], dim=0)
            activity = activity_regularizer(
                joined,
                base["regularization"]["target_rate"],
                base["regularization"]["balance_weight"],
            )
            loss = semantic + base["regularization"]["rate_weight"] * activity
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        final_loss = sum(losses) / len(losses)
    return model, final_loss


def split_spikes(joined: torch.Tensor):
    batch = joined.shape[0] // 3
    shaped = joined.reshape(batch, 3, joined.shape[1], joined.shape[2])
    return shaped[:, 0], shaped[:, 1], shaped[:, 2]


def corrupt_triplet(a, p, n, mode, amount, generator):
    # Interleave como [a,p,n] por tripleta para reutilizar split_spikes.
    joined = torch.stack([a, p, n], dim=1).flatten(0, 1)
    if mode == "local":
        corrupted = local_time_shuffle(joined, amount, generator)
    elif mode == "global":
        corrupted = global_time_shuffle(joined, generator)
    else:
        raise ValueError(mode)
    assert_preserves_per_neuron_counts(joined, corrupted)
    return split_spikes(corrupted)


def mean_corrupted_sta(a, p, n, taus, weights, mode, amount, trials, seed):
    values = []
    for trial in range(trials):
        generator = torch.Generator(device=a.device).manual_seed(seed + trial)
        ca, cp, cn = corrupt_triplet(a, p, n, mode, amount, generator)
        values.append(triplet_metrics(ca, cp, cn, taus, weights).temporal_sta)
    return float(np.mean(values))


def metrics_for_subset(a, p, n, mask, taus, weights, suite, seed):
    a, p, n = a[mask], p[mask], n[mask]
    original = triplet_metrics(a, p, n, taus, weights)
    result = {
        "examples": int(mask.sum().item()),
        "original_van_rossum_sta": original.temporal_sta,
        "rate_only_sta": original.rate_sta,
        "semantic_positive_distance": original.mean_positive_distance,
        "negative_distance": original.mean_negative_distance,
        "spike_rate": original.spike_rate,
        "active_neurons": original.active_neurons,
        "local_shuffle": {},
    }
    for radius in suite["local_radii"]:
        result["local_shuffle"][f"radius_{radius}"] = mean_corrupted_sta(
            a, p, n, taus, weights, "local", radius, suite["shuffle_trials"], seed + 1000 * radius
        )
    result["global_shuffle_sta"] = mean_corrupted_sta(
        a, p, n, taus, weights, "global", 0, suite["shuffle_trials"], seed + 9000
    )
    return result


@torch.no_grad()
def evaluate_seed(model, encoder, validation, base, suite, seed):
    model.eval()
    a, p, n = encode_triplet_batch(model, encoder, validation, torch.device("cpu"))
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    kinds = [item.kind for item in validation]
    result = {}
    subsets = {
        "all": torch.ones(len(validation), dtype=torch.bool),
        "random_negative": torch.tensor([kind == "random_negative" for kind in kinds]),
        "hard_orthographic": torch.tensor([kind == "hard_orthographic" for kind in kinds]),
    }
    for name, mask in subsets.items():
        result[name] = metrics_for_subset(a, p, n, mask, taus, weights, suite, seed)

    # Diagnostico por capa sobre el mismo validation set.
    texts = []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
    layers = model(encoder.encode(texts), return_all=True)
    result["layers"] = []
    stimulus_end = encoder.max_chars * encoder.char_steps
    for index, layer in enumerate(layers, start=1):
        la, lp, ln = split_spikes(layer)
        layer_metric = triplet_metrics(la, lp, ln, taus, weights)
        result["layers"].append(
            {
                "layer": index,
                "sta": layer_metric.temporal_sta,
                "stimulus_window_rate": layer[:, :stimulus_end].mean().item(),
                "post_window_rate": layer[:, stimulus_end:].mean().item(),
                "active_neurons": layer_metric.active_neurons,
            }
        )
    return result


def bootstrap_interval(values, samples, seed=20260928):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
    }


def aggregate(seed_results, suite):
    paths = {
        "original_van_rossum_sta": lambda r: r["evaluation"]["all"]["original_van_rossum_sta"],
        "local_radius_1_sta": lambda r: r["evaluation"]["all"]["local_shuffle"]["radius_1"],
        "local_radius_3_sta": lambda r: r["evaluation"]["all"]["local_shuffle"]["radius_3"],
        "global_shuffle_sta": lambda r: r["evaluation"]["all"]["global_shuffle_sta"],
        "rate_only_sta": lambda r: r["evaluation"]["all"]["rate_only_sta"],
        "random_negative_original_sta": lambda r: r["evaluation"]["random_negative"]["original_van_rossum_sta"],
        "hard_orthographic_original_sta": lambda r: r["evaluation"]["hard_orthographic"]["original_van_rossum_sta"],
        "semantic_positive_distance": lambda r: r["evaluation"]["all"]["semantic_positive_distance"],
        "random_negative_distance": lambda r: r["evaluation"]["random_negative"]["negative_distance"],
        "hard_orthographic_negative_distance": lambda r: r["evaluation"]["hard_orthographic"]["negative_distance"],
    }
    for subset in ("random_negative", "hard_orthographic"):
        paths.update(
            {
                f"{subset}_local_radius_1_sta": lambda r, s=subset: r["evaluation"][s]["local_shuffle"]["radius_1"],
                f"{subset}_local_radius_3_sta": lambda r, s=subset: r["evaluation"][s]["local_shuffle"]["radius_3"],
                f"{subset}_global_shuffle_sta": lambda r, s=subset: r["evaluation"][s]["global_shuffle_sta"],
                f"{subset}_rate_only_sta": lambda r, s=subset: r["evaluation"][s]["rate_only_sta"],
            }
        )
    for layer_index in range(3):
        layer_number = layer_index + 1
        paths.update(
            {
                f"layer_{layer_number}_sta": lambda r, i=layer_index: r["evaluation"]["layers"][i]["sta"],
                f"layer_{layer_number}_stimulus_rate": lambda r, i=layer_index: r["evaluation"]["layers"][i]["stimulus_window_rate"],
                f"layer_{layer_number}_post_rate": lambda r, i=layer_index: r["evaluation"]["layers"][i]["post_window_rate"],
            }
        )
    aggregated = {
        name: bootstrap_interval([getter(result) for result in seed_results], suite["bootstrap_samples"])
        for name, getter in paths.items()
    }
    paired_differences = {
        "original_minus_local_radius_1": lambda r: (
            r["evaluation"]["all"]["original_van_rossum_sta"]
            - r["evaluation"]["all"]["local_shuffle"]["radius_1"]
        ),
        "original_minus_local_radius_3": lambda r: (
            r["evaluation"]["all"]["original_van_rossum_sta"]
            - r["evaluation"]["all"]["local_shuffle"]["radius_3"]
        ),
        "original_minus_global": lambda r: (
            r["evaluation"]["all"]["original_van_rossum_sta"]
            - r["evaluation"]["all"]["global_shuffle_sta"]
        ),
        "original_minus_rate_only": lambda r: (
            r["evaluation"]["all"]["original_van_rossum_sta"]
            - r["evaluation"]["all"]["rate_only_sta"]
        ),
    }
    aggregated["paired_differences"] = {
        name: bootstrap_interval([getter(result) for result in seed_results], suite["bootstrap_samples"])
        for name, getter in paired_differences.items()
    }
    return aggregated


def build_summary(seed_results, suite, model_parameters, elapsed_seconds):
    validation = make_fixed_validation(suite["validation_random_size"])
    return {
        "experiment": "E1.1-temporal-controls",
        "model_parameters": model_parameters,
        "seeds": suite["seeds"],
        "validation_examples": len(validation),
        "validation_composition": {
            kind: sum(item.kind == kind for item in validation)
            for kind in ("random_negative", "hard_orthographic")
        },
        "shuffle_trials_per_seed": suite["shuffle_trials"],
        "elapsed_seconds": elapsed_seconds,
        "aggregate": aggregate(seed_results, suite),
        "interpretation_policy": (
            "Original > global/rate de forma reproducible apoya H3; un empate no la apoya. "
            "Los intervalos con 5 seeds son descriptivos y no sustituyen replicacion."
        ),
    }


def run(config_path: str | Path = "configs/e11.toml", epochs_override=None):
    suite, base = load_configs(Path(config_path))
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(suite["validation_random_size"])
    output_dir = Path(suite["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs_override or base["run"]["epochs"]
    seed_results = []
    started = time.perf_counter()
    for seed in suite["seeds"]:
        training = make_e11_training_triplets(seed, validation, suite["training_repeats"])
        seed_started = time.perf_counter()
        model, final_loss = train_one(seed, base, encoder, training, epochs)
        evaluation = evaluate_seed(model, encoder, validation, base, suite, seed)
        result = {
            "seed": seed,
            "train_examples": len(training),
            "validation_examples": len(validation),
            "epochs": epochs,
            "final_loss": final_loss,
            "elapsed_seconds": time.perf_counter() - seed_started,
            "evaluation": evaluation,
        }
        seed_results.append(result)
        (output_dir / f"seed_{seed}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        torch.save(model.state_dict(), output_dir / f"seed_{seed}.pt")
        all_metrics = evaluation["all"]
        print(
            f"seed={seed} original={all_metrics['original_van_rossum_sta']:.3f} "
            f"local1={all_metrics['local_shuffle']['radius_1']:.3f} "
            f"local3={all_metrics['local_shuffle']['radius_3']:.3f} "
            f"global={all_metrics['global_shuffle_sta']:.3f} "
            f"rate={all_metrics['rate_only_sta']:.3f}"
        )
    summary = build_summary(
        seed_results,
        suite,
        sum(parameter.numel() for parameter in model.parameters()),
        time.perf_counter() - started,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def summarize_existing(config_path: str | Path = "configs/e11.toml"):
    suite, base = load_configs(Path(config_path))
    output_dir = Path(suite["output_dir"])
    seed_results = [
        json.loads((output_dir / f"seed_{seed}.json").read_text(encoding="utf-8"))
        for seed in suite["seeds"]
    ]
    encoder = EventEncoder(**base["input"])
    model = HierarchicalSNN(encoder.channels, **base["model"])
    previous = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    summary = build_summary(
        seed_results,
        suite,
        sum(parameter.numel() for parameter in model.parameters()),
        previous.get("elapsed_seconds"),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def reevaluate_existing(config_path: str | Path = "configs/e11.toml"):
    suite, base = load_configs(Path(config_path))
    output_dir = Path(suite["output_dir"])
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(suite["validation_random_size"])
    for seed in suite["seeds"]:
        path = output_dir / f"seed_{seed}.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        model = HierarchicalSNN(encoder.channels, **base["model"])
        model.load_state_dict(
            torch.load(output_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        )
        result["evaluation"] = evaluate_seed(model, encoder, validation, base, suite, seed)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"reevaluated seed={seed}")
    return summarize_existing(config_path)


def main():
    parser = argparse.ArgumentParser(description="E1.1: controles temporales multisemilla")
    parser.add_argument("--config", default="configs/e11.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--reevaluate-only", action="store_true")
    args = parser.parse_args()
    if args.reevaluate_only:
        reevaluate_existing(args.config)
    elif args.summarize_only:
        summarize_existing(args.config)
    else:
        run(args.config, args.epochs)


if __name__ == "__main__":
    main()
