from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
import torch

from .data import EventEncoder, make_fixed_validation
from .e11 import bootstrap_interval, load_configs
from .model import HierarchicalSNN


def rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks promedio para empates, suficiente para Spearman sin scipy."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(x, y, rank=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if rank:
        x, y = rankdata(x), rankdata(y)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


@torch.no_grad()
def evaluate_seed(seed, base, e11, checkpoint_dir):
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(e11["validation_random_size"])
    texts, hard_text_mask = [], []
    for item in validation:
        texts.extend([item.anchor, item.positive, item.negative])
        hard_text_mask.extend([item.kind == "hard_orthographic"] * 3)
    events = encoder.encode(texts)
    initial = HierarchicalSNN(
        encoder.channels, **base["model"], tau_mode="heterogeneous_learnable"
    )
    final = HierarchicalSNN(
        encoder.channels, **base["model"], tau_mode="heterogeneous_learnable"
    )
    final.load_state_dict(
        torch.load(checkpoint_dir / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    )
    final.eval()
    layers = final(events, return_all=True)
    stimulus_end = encoder.max_chars * encoder.char_steps
    hard_mask = torch.tensor(hard_text_mask, dtype=torch.bool)
    result = {"seed": seed, "layers": []}
    for layer_index, activity in enumerate(layers):
        beta_initial = torch.sigmoid(initial.layers[layer_index].beta_logit).numpy()
        beta_final = torch.sigmoid(final.layers[layer_index].beta_logit).numpy()
        delta = beta_final - beta_initial
        firing = activity.mean(dim=(0, 1)).numpy()
        post_firing = activity[:, stimulus_end:].mean(dim=(0, 1)).numpy()
        hard_firing = activity[hard_mask].mean(dim=(0, 1)).numpy()
        top_count = max(1, len(hard_firing) // 4)
        order = np.argsort(hard_firing)
        bottom, top = order[:top_count], order[-top_count:]
        result["layers"].append(
            {
                "layer": layer_index + 1,
                "initial_beta": distribution(beta_initial),
                "final_beta": distribution(beta_final),
                "delta_beta": distribution(delta),
                "delta_variance": float(np.var(delta)),
                "mean_absolute_delta": float(np.abs(delta).mean()),
                "correlations": {
                    "abs_delta_vs_firing_pearson": correlation(np.abs(delta), firing),
                    "abs_delta_vs_firing_spearman": correlation(np.abs(delta), firing, rank=True),
                    "beta_vs_post_firing_pearson": correlation(beta_final, post_firing),
                    "beta_vs_post_firing_spearman": correlation(beta_final, post_firing, rank=True),
                    "beta_vs_hard_firing_pearson": correlation(beta_final, hard_firing),
                    "beta_vs_hard_firing_spearman": correlation(beta_final, hard_firing, rank=True),
                    "abs_delta_vs_hard_firing_spearman": correlation(
                        np.abs(delta), hard_firing, rank=True
                    ),
                },
                "hard_activity_quartiles": {
                    "bottom_beta_mean": float(beta_final[bottom].mean()),
                    "top_beta_mean": float(beta_final[top].mean()),
                    "bottom_abs_delta_mean": float(np.abs(delta[bottom]).mean()),
                    "top_abs_delta_mean": float(np.abs(delta[top]).mean()),
                    "bottom_firing_mean": float(hard_firing[bottom].mean()),
                    "top_firing_mean": float(hard_firing[top].mean()),
                },
                "per_neuron": {
                    "initial_beta": beta_initial.tolist(),
                    "final_beta": beta_final.tolist(),
                    "delta_beta": delta.tolist(),
                    "firing_rate": firing.tolist(),
                    "post_firing_rate": post_firing.tolist(),
                    "hard_firing_rate": hard_firing.tolist(),
                },
            }
        )
    return result


def aggregate(seed_results, samples):
    result = []
    for layer_index in range(3):
        row = {"layer": layer_index + 1}
        paths = {
            "initial_std": lambda x: x["initial_beta"]["std"],
            "final_std": lambda x: x["final_beta"]["std"],
            "delta_mean": lambda x: x["delta_beta"]["mean"],
            "delta_std": lambda x: x["delta_beta"]["std"],
            "delta_variance": lambda x: x["delta_variance"],
            "mean_absolute_delta": lambda x: x["mean_absolute_delta"],
        }
        for name, getter in paths.items():
            values = [getter(seed["layers"][layer_index]) for seed in seed_results]
            row[name] = bootstrap_interval(values, samples)
        row["correlations"] = {}
        correlation_names = seed_results[0]["layers"][layer_index]["correlations"].keys()
        for name in correlation_names:
            values = [
                seed["layers"][layer_index]["correlations"][name]
                for seed in seed_results
                if seed["layers"][layer_index]["correlations"][name] is not None
            ]
            row["correlations"][name] = bootstrap_interval(values, samples)
        row["hard_activity_quartiles"] = {}
        quartile_names = seed_results[0]["layers"][layer_index]["hard_activity_quartiles"].keys()
        for name in quartile_names:
            values = [
                seed["layers"][layer_index]["hard_activity_quartiles"][name]
                for seed in seed_results
            ]
            row["hard_activity_quartiles"][name] = bootstrap_interval(values, samples)
        result.append(row)
    return result


def run(config_path="configs/e12b.toml"):
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    checkpoint_dir = Path(config["output_dir"]) / "heterogeneous_learnable"
    output_dir = Path(config["output_dir"])
    seed_results = []
    for seed in e11["seeds"]:
        result = evaluate_seed(seed, base, e11, checkpoint_dir)
        seed_results.append(result)
        (output_dir / f"beta_seed_{seed}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {
        "experiment": "E1.2B-beta-specialization-diagnostics",
        "note": "Initial beta is already heterogeneous within each layer.",
        "seeds": e11["seeds"],
        "aggregate": aggregate(seed_results, config["bootstrap_samples"]),
    }
    (output_dir / "beta_diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Diagnostico de especializacion de beta")
    parser.add_argument("--config", default="configs/e12b.toml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

