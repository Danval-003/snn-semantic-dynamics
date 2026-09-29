from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import torch

from .data import EventEncoder, make_fixed_validation
from .e11 import bootstrap_interval
from .factorial import CELL_NAMES, audit, load_pairs
from .models import make_model
from .provenance import write_manifest
from .readouts import READOUTS, extract_readouts


def vector_distance(left, right):
    return (left - right).square().mean(dim=1)


def triplet_statistics(vectors, triplets):
    shaped = vectors.reshape(len(triplets), 3, -1)
    positive = vector_distance(shaped[:, 0], shaped[:, 1])
    negative = vector_distance(shaped[:, 0], shaped[:, 2])
    hard = torch.tensor(
        [row.kind == "hard_orthographic" for row in triplets], device=vectors.device
    )
    return {
        "global_sta": (positive < negative).float().mean().item(),
        "hard_sta": (positive[hard] < negative[hard]).float().mean().item(),
        "global_margin": (negative - positive).mean().item(),
        "hard_margin": (negative[hard] - positive[hard]).mean().item(),
    }


def factorial_statistics(vectors, rows):
    distances = vector_distance(vectors[0::2], vectors[1::2])
    names = [CELL_NAMES[(row["orthography"], row["semantics"])] for row in rows]
    means = {
        cell: distances[torch.tensor(
            [name == cell for name in names], device=vectors.device
        )].mean().item()
        for cell in CELL_NAMES.values()
    }
    a = means["orth+_semantic+"]
    b = means["orth+_semantic-"]
    c = means["orth-_semantic+"]
    d = means["orth-_semantic-"]
    semantic = ((b + d) - (a + c)) / 2
    orthographic = ((c + d) - (a + b)) / 2
    grand = (a + b + c + d) / 4
    return {
        "cells": means,
        "semantic_effect": semantic,
        "orthographic_effect": orthographic,
        "abstraction_index": semantic - orthographic,
        "abstraction_index_relative": (semantic - orthographic) / (grand + 1e-8),
    }


def encode_texts(encoder, texts, evaluation_steps):
    return encoder.encode_explicit(texts, settling_steps=evaluation_steps)


def parity_check(model, encoder, texts):
    legacy_events = encoder.encode(texts)
    explicit = encoder.encode_explicit(
        texts, settling_steps=encoder.post_steps, minimum_steps=encoder.total_steps
    )
    if not torch.equal(legacy_events, explicit.events):
        return {"events_equal": False, "states_equal": False, "metrics_equal": False}
    legacy_layers = model(legacy_events, return_all=True)
    explicit_layers = model(
        explicit.events, return_all=True, stimulus_ends=explicit.stimulus_ends,
        settling_intervention="natural",
    )
    states_equal = all(
        torch.equal(left, right) for left, right in zip(legacy_layers, explicit_layers, strict=True)
    )
    legacy_vectors = legacy_layers[-1].sum(1)
    explicit_vectors = explicit_layers[-1].sum(1)
    return {
        "events_equal": True,
        "states_equal": states_equal,
        "metrics_equal": torch.equal(legacy_vectors, explicit_vectors),
    }


@torch.no_grad()
def evaluate_seed(model, encoder, validation, pair_sets, config, intervention):
    maximum = max(config["offsets"])
    triplet_texts = [
        text for row in validation for text in (row.anchor, row.positive, row.negative)
    ]
    triplet_batch = encode_texts(encoder, triplet_texts, maximum)
    triplet_layers = model(
        triplet_batch.events, return_all=True,
        stimulus_ends=triplet_batch.stimulus_ends,
        settling_intervention=intervention,
    )
    pair_activity = {}
    for name, rows in pair_sets.items():
        texts = [text for row in rows for text in (row["left"], row["right"])]
        encoded = encode_texts(encoder, texts, maximum)
        pair_activity[name] = (
            model(
                encoded.events, return_all=True,
                stimulus_ends=encoded.stimulus_ends,
                settling_intervention=intervention,
            ),
            encoded.stimulus_ends,
        )

    layers = []
    for layer_index, triplet_activity in enumerate(triplet_layers):
        layer = {"layer": layer_index + 1, "offsets": []}
        for offset in config["offsets"]:
            triplet_views = extract_readouts(
                triplet_activity, triplet_batch.stimulus_ends, offset, config["local_width"]
            )
            readouts = {}
            for readout in READOUTS:
                row = {"triplets": triplet_statistics(triplet_views[readout], validation)}
                row["pair_sets"] = {}
                for name, pairs in pair_sets.items():
                    activities, ends = pair_activity[name]
                    pair_views = extract_readouts(
                        activities[layer_index], ends, offset, config["local_width"]
                    )
                    row["pair_sets"][name] = factorial_statistics(pair_views[readout], pairs)
                readouts[readout] = row
            layer["offsets"].append({"offset": offset, "readouts": readouts})
        layers.append(layer)
    return {"intervention": intervention, "layers": layers}


def aggregate(results, config):
    fields = ("global_sta", "hard_sta", "global_margin", "hard_margin")
    output = {"models": {}}
    for model_kind in config["models"]:
        output["models"][model_kind] = {}
        for intervention in config["interventions"]:
            selected = [
                row for row in results
                if row["model"] == model_kind and row["intervention"] == intervention
            ]
            layers = []
            for layer_index in range(3):
                layer = {"layer": layer_index + 1, "offsets": []}
                for offset_index, offset in enumerate(config["offsets"]):
                    target = {"offset": offset, "readouts": {}}
                    for readout in READOUTS:
                        sources = [
                            row["evaluation"]["layers"][layer_index]["offsets"][offset_index]
                            ["readouts"][readout] for row in selected
                        ]
                        view = {
                            "triplets": {
                                field: bootstrap_interval(
                                    [source["triplets"][field] for source in sources],
                                    config["bootstrap_samples"],
                                ) for field in fields
                            },
                            "pair_sets": {},
                        }
                        for name in config["pair_sets"]:
                            view["pair_sets"][name] = {
                                "abstraction_index_relative": bootstrap_interval(
                                    [source["pair_sets"][name]["abstraction_index_relative"]
                                     for source in sources],
                                    config["bootstrap_samples"],
                                )
                            }
                        target["readouts"][readout] = view
                    layer["offsets"].append(target)
                layers.append(layer)
            output["models"][model_kind][intervention] = {"layers": layers}
    lookup = {
        (row["model"], row["seed"], row["intervention"]): row for row in results
    }
    output["paired_changes"] = {}
    layer_index = 2
    readout = "full_trajectory"
    for model_kind in config["models"]:
        output["paired_changes"][model_kind] = {
            "natural_change_from_t0": {}, "natural_minus_intervention": {}
        }
        for offset_index, offset in enumerate(config["offsets"]):
            def source(seed, intervention, index=offset_index):
                return lookup[(model_kind, seed, intervention)]["evaluation"]["layers"] \
                    [layer_index]["offsets"][index]["readouts"][readout]

            baseline_index = config["offsets"].index(0)
            changes = {}
            for field in ("global_sta", "hard_sta"):
                changes[field] = bootstrap_interval([
                    source(seed, "natural")["triplets"][field]
                    - source(seed, "natural", baseline_index)["triplets"][field]
                    for seed in config["seeds"]
                ], config["bootstrap_samples"])
            for pair_name in config["pair_sets"]:
                changes[f"{pair_name}_A3"] = bootstrap_interval([
                    source(seed, "natural")["pair_sets"][pair_name]
                    ["abstraction_index_relative"]
                    - source(seed, "natural", baseline_index)["pair_sets"][pair_name]
                    ["abstraction_index_relative"]
                    for seed in config["seeds"]
                ], config["bootstrap_samples"])
            output["paired_changes"][model_kind]["natural_change_from_t0"][str(offset)] = changes

        for intervention in config["interventions"]:
            if intervention == "natural":
                continue
            output["paired_changes"][model_kind]["natural_minus_intervention"][intervention] = {}
            for offset_index, offset in enumerate(config["offsets"]):
                differences = {}
                for field in ("global_sta", "hard_sta"):
                    differences[field] = bootstrap_interval([
                        source(seed, "natural", offset_index)["triplets"][field]
                        - source(seed, intervention, offset_index)["triplets"][field]
                        for seed in config["seeds"]
                    ], config["bootstrap_samples"])
                for pair_name in config["pair_sets"]:
                    differences[f"{pair_name}_A3"] = bootstrap_interval([
                        source(seed, "natural", offset_index)["pair_sets"][pair_name]
                        ["abstraction_index_relative"]
                        - source(seed, intervention, offset_index)["pair_sets"][pair_name]
                        ["abstraction_index_relative"]
                        for seed in config["seeds"]
                    ], config["bootstrap_samples"])
                output["paired_changes"][model_kind]["natural_minus_intervention"] \
                    [intervention][str(offset)] = differences
    return output


def run(config_path="configs/mechanistic.toml"):
    config_path = Path(config_path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    with Path(config["base_config"]).open("rb") as handle:
        base = tomllib.load(handle)
    encoder = EventEncoder(**base["input"])
    validation = make_fixed_validation(config["validation_random_size"])
    pair_sets = {}
    audits = {}
    for name, path in config["pair_sets"].items():
        rows = load_pairs(Path(path))
        _, audits[name] = audit(rows)
        pair_sets[name] = rows
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    parity = []
    outputs = []
    parity_texts = ["perro", "automóvil", "se detuvo"]
    for model_kind in config["models"]:
        for seed in config["seeds"]:
            model = make_model(model_kind, encoder.channels, base["model"])
            checkpoint = Path(config["checkpoint_dirs"][model_kind]) / f"seed_{seed}.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            check = parity_check(model, encoder, parity_texts)
            parity.append({"model": model_kind, "seed": seed, **check})
            if not all(check.values()):
                raise RuntimeError(f"Falló paridad legacy para {model_kind} seed={seed}: {check}")
            for intervention in config["interventions"]:
                evaluation = evaluate_seed(
                    model, encoder, validation, pair_sets, config, intervention
                )
                result = {
                    "model": model_kind, "seed": seed,
                    "intervention": intervention, "evaluation": evaluation,
                }
                results.append(result)
                path = output_dir / f"{model_kind}_seed_{seed}_{intervention}.json"
                path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                outputs.append(path)
                primary = evaluation["layers"][-1]["offsets"]
                end = next(row for row in primary if row["offset"] == 12)
                stats = end["readouts"]["full_trajectory"]["triplets"]
                print(
                    f"{model_kind} seed={seed} {intervention} "
                    f"STA={stats['global_sta']:.3f} hard={stats['hard_sta']:.3f}"
                )
    summary = {
        "experiment": "E1.5-post-stimulus-recurrent-settling",
        "analysis_only": True,
        "readouts": list(READOUTS),
        "temporal_invariant": "offsets are relative to each example's final real event",
        "parity": parity,
        "dataset_audits": audits,
        "config": config,
        "aggregate": aggregate(results, config),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs.append(summary_path)
    write_manifest(
        output_dir,
        experiment=summary["experiment"], config_path=config_path,
        seeds=config["seeds"], datasets=list(config["pair_sets"].values()),
        outputs=outputs,
        checkpoints=[
            Path(config["checkpoint_dirs"][model_kind]) / f"seed_{seed}.pt"
            for model_kind in config["models"] for seed in config["seeds"]
        ],
        extra={
            "checkpoints": config["checkpoint_dirs"],
            "legacy_parity_passed": all(all(
                row[key] for key in ("events_equal", "states_equal", "metrics_equal")
            ) for row in parity),
        },
    )
    return summary


def summarize_existing(config_path="configs/mechanistic.toml"):
    """Recompute all aggregate estimands without loading checkpoints."""
    config_path = Path(config_path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output_dir = Path(config["output_dir"])
    summary_path = output_dir / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    results = []
    outputs = []
    for model_kind in config["models"]:
        for seed in config["seeds"]:
            for intervention in config["interventions"]:
                path = output_dir / f"{model_kind}_seed_{seed}_{intervention}.json"
                results.append(json.loads(path.read_text(encoding="utf-8")))
                outputs.append(path)
    previous["aggregate"] = aggregate(results, config)
    summary_path.write_text(
        json.dumps(previous, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    outputs.append(summary_path)
    write_manifest(
        output_dir,
        experiment=previous["experiment"], config_path=config_path,
        seeds=config["seeds"], datasets=list(config["pair_sets"].values()),
        outputs=outputs,
        checkpoints=[
            Path(config["checkpoint_dirs"][model_kind]) / f"seed_{seed}.pt"
            for model_kind in config["models"] for seed in config["seeds"]
        ],
        extra={
            "checkpoints": config["checkpoint_dirs"],
            "legacy_parity_passed": all(all(
                row[key] for key in ("events_equal", "states_equal", "metrics_equal")
            ) for row in previous["parity"]),
            "summary_recomputed_without_checkpoints": True,
        },
    )
    return previous


def main():
    parser = argparse.ArgumentParser(description="E1.5: settling mecanístico congelado")
    parser.add_argument("--config", default="configs/mechanistic.toml")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize_existing(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
