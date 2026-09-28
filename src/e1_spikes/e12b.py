from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

import torch

from .data import EventEncoder, make_e11_training_triplets, make_fixed_validation
from .e11 import bootstrap_interval, load_configs, split_spikes, train_one
from .e12a import distance_statistics
from .model import HierarchicalSNN


def load_config(path: Path):
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    return config, e11, base


def beta_summary(model):
    layers = []
    for layer_number, layer in enumerate(model.layers, start=1):
        beta = torch.sigmoid(layer.beta_logit.detach())
        effective_tau = -1.0 / torch.log(beta)
        layers.append(
            {
                "layer": layer_number,
                "learnable": isinstance(layer.beta_logit, torch.nn.Parameter),
                "beta_mean": beta.mean().item(),
                "beta_std": beta.std(unbiased=False).item(),
                "beta_min": beta.min().item(),
                "beta_max": beta.max().item(),
                "effective_tau_steps_mean": effective_tau.mean().item(),
                "effective_tau_steps_min": effective_tau.min().item(),
                "effective_tau_steps_max": effective_tau.max().item(),
            }
        )
    return layers


def masks_for(validation):
    kinds = [item.kind for item in validation]
    return {
        "all": torch.ones(len(validation), dtype=torch.bool),
        "random_negative": torch.tensor([kind == "random_negative" for kind in kinds]),
        "hard_orthographic": torch.tensor([kind == "hard_orthographic" for kind in kinds]),
    }


@torch.no_grad()
def evaluate(model, encoder, validation, base):
    texts = []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
    layers = model(encoder.encode(texts), return_all=True)
    masks = masks_for(validation)
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    stimulus_end = encoder.max_chars * encoder.char_steps
    result = {"subsets": {}, "activity": [], "beta": beta_summary(model)}
    for subset, mask in masks.items():
        result["subsets"][subset] = []
        for layer_number, layer in enumerate(layers, start=1):
            a, p, n = split_spikes(layer)
            result["subsets"][subset].append(
                {
                    "layer": layer_number,
                    "van_rossum": distance_statistics(
                        a[mask], p[mask], n[mask], taus, weights
                    ),
                    "rate_only": distance_statistics(
                        a[mask], p[mask], n[mask], rate_only=True
                    ),
                }
            )
    for layer_number, layer in enumerate(layers, start=1):
        per_neuron = layer.mean(dim=(0, 1))
        result["activity"].append(
            {
                "layer": layer_number,
                "stimulus_rate": layer[:, :stimulus_end].mean().item(),
                "post_rate": layer[:, stimulus_end:].mean().item(),
                "active_neurons": (per_neuron > 0).float().mean().item(),
            }
        )
    return result


def aggregate_variant(seed_results, samples):
    result = {"subsets": {}, "activity": [], "beta": []}
    for subset in ("all", "random_negative", "hard_orthographic"):
        result["subsets"][subset] = []
        for layer_index in range(3):
            row = {"layer": layer_index + 1}
            for metric in ("van_rossum", "rate_only"):
                row[metric] = {}
                for field in ("sta", "positive_distance", "negative_distance", "margin"):
                    values = [
                        seed_result["evaluation"]["subsets"][subset][layer_index][metric][field]
                        for seed_result in seed_results
                    ]
                    row[metric][field] = bootstrap_interval(values, samples)
            result["subsets"][subset].append(row)
    for layer_index in range(3):
        row = {"layer": layer_index + 1}
        for field in ("stimulus_rate", "post_rate", "active_neurons"):
            values = [
                seed_result["evaluation"]["activity"][layer_index][field]
                for seed_result in seed_results
            ]
            row[field] = bootstrap_interval(values, samples)
        result["activity"].append(row)
        beta_row = {"layer": layer_index + 1}
        for field in (
            "beta_mean",
            "beta_std",
            "beta_min",
            "beta_max",
            "effective_tau_steps_mean",
        ):
            values = [
                seed_result["evaluation"]["beta"][layer_index][field]
                for seed_result in seed_results
            ]
            beta_row[field] = bootstrap_interval(values, samples)
        result["beta"].append(beta_row)
    return result


def paired_comparisons(all_results, variants, samples):
    reference = "heterogeneous_learnable"
    comparisons = {}
    for variant in variants:
        if variant == reference:
            continue
        comparisons[f"{reference}_minus_{variant}"] = {}
        for target_name, subset, layer_index, metric, field in (
            ("global_sta", "all", 2, "van_rossum", "sta"),
            ("hard_sta", "hard_orthographic", 2, "van_rossum", "sta"),
            ("hard_margin_l3", "hard_orthographic", 2, "van_rossum", "margin"),
        ):
            values = []
            by_seed_reference = {row["seed"]: row for row in all_results[reference]}
            by_seed_variant = {row["seed"]: row for row in all_results[variant]}
            for seed in sorted(by_seed_reference):
                ref_value = by_seed_reference[seed]["evaluation"]["subsets"][subset][layer_index][metric][field]
                variant_value = by_seed_variant[seed]["evaluation"]["subsets"][subset][layer_index][metric][field]
                values.append(ref_value - variant_value)
            comparisons[f"{reference}_minus_{variant}"][target_name] = bootstrap_interval(
                values, samples
            )
        post_values = []
        by_seed_reference = {row["seed"]: row for row in all_results[reference]}
        by_seed_variant = {row["seed"]: row for row in all_results[variant]}
        for seed in sorted(by_seed_reference):
            post_values.append(
                by_seed_reference[seed]["evaluation"]["activity"][2]["post_rate"]
                - by_seed_variant[seed]["evaluation"]["activity"][2]["post_rate"]
            )
        comparisons[f"{reference}_minus_{variant}"]["post_rate_l3"] = bootstrap_interval(
            post_values, samples
        )
    return comparisons


def run(config_path: str | Path = "configs/e12b.toml", epochs_override=None, seeds_override=None):
    config, e11, base = load_config(Path(config_path))
    seeds = seeds_override or e11["seeds"]
    epochs = epochs_override or base["run"]["epochs"]
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(e11["validation_random_size"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {variant: [] for variant in config["variants"]}
    started = time.perf_counter()
    for variant in config["variants"]:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            training = make_e11_training_triplets(seed, validation, e11["training_repeats"])
            run_started = time.perf_counter()
            model, final_loss = train_one(
                seed, base, encoder, training, epochs, tau_mode=variant
            )
            evaluation = evaluate(model, encoder, validation, base)
            result = {
                "variant": variant,
                "seed": seed,
                "epochs": epochs,
                "train_examples": len(training),
                "final_loss": final_loss,
                "elapsed_seconds": time.perf_counter() - run_started,
                "total_state_values": sum(value.numel() for value in model.state_dict().values()),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "evaluation": evaluation,
            }
            all_results[variant].append(result)
            (variant_dir / f"seed_{seed}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            torch.save(model.state_dict(), variant_dir / f"seed_{seed}.pt")
            final = evaluation["subsets"]["all"][2]["van_rossum"]["sta"]
            hard = evaluation["subsets"]["hard_orthographic"][2]["van_rossum"]["sta"]
            print(
                f"variant={variant} seed={seed} STA={final:.3f} "
                f"hard={hard:.3f} elapsed={result['elapsed_seconds']:.1f}s"
            )

    aggregate = {
        variant: aggregate_variant(results, config["bootstrap_samples"])
        for variant, results in all_results.items()
    }
    summary = {
        "experiment": "E1.2B-neuronal-timescale-ablation",
        "scope": (
            "Ablates membrane beta only. Recurrence, adaptation and adaptation decay "
            "remain identical and learnable across variants."
        ),
        "seeds": seeds,
        "epochs": epochs,
        "validation_examples": len(validation),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate,
        "paired_comparisons": paired_comparisons(
            all_results, config["variants"], config["bootstrap_samples"]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="E1.2B: escalas temporales neuronales")
    parser.add_argument("--config", default="configs/e12b.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.config, args.epochs, args.seeds)


if __name__ == "__main__":
    main()
