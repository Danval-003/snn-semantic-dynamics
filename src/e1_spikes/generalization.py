from __future__ import annotations

import argparse
import itertools
import json
import random
import time
import tomllib
from pathlib import Path

import torch

from .ann_baseline import train_one as train_ann
from .data import CONCEPTS, EventEncoder, Triplet
from .e11 import bootstrap_interval, load_configs, split_spikes, train_one as train_snn
from .e12a import distance_statistics


HELD_RELATIONS = (
    ("perro", "can"),
    ("gato", "felino"),
    ("automóvil", "carro"),
    ("coche", "vehículo"),
    ("rápido", "veloz"),
    ("casa", "hogar"),
    ("lluvia", "nube"),
    ("pino", "árbol"),
    ("paró", "se detuvo"),
)


CONTEXT_GROUPS = {
    "animal": {
        "known": ("perro", "gato"),
        "new": ("ornitorrinco", "dromedario"),
        "patterns": ("{word} tiene patas", "{word} come alimento"),
    },
    "vehicle": {
        "known": ("carro", "coche"),
        "new": ("tranvía", "bicicleta"),
        "patterns": ("{word} tiene ruedas", "{word} lleva personas"),
    },
    "nature": {
        "known": ("pino", "árbol"),
        "new": ("ciprés", "roble"),
        "patterns": ("{word} tiene ramas", "{word} crece alto"),
    },
    "dwelling": {
        "known": ("casa", "hogar"),
        "new": ("choza", "cabaña"),
        "patterns": ("{word} tiene techo", "{word} sirve vivienda"),
    },
}


def unordered_pair(left, right):
    return tuple(sorted((left, right)))


def build_relation_data(seed=20260928):
    held = {unordered_pair(*pair) for pair in HELD_RELATIONS}
    groups = list(CONCEPTS.values())
    training = []
    for group_index, words in enumerate(groups):
        negatives = [word for index, group in enumerate(groups) if index != group_index for word in group]
        for left, right in itertools.permutations(words, 2):
            if unordered_pair(left, right) in held:
                continue
            # Dos negativos deterministas por relacion mantienen el conjunto pequeno.
            offset = sum(ord(char) for char in left + right) % len(negatives)
            for step in (0, len(negatives) // 2):
                training.append(
                    Triplet(left, right, negatives[(offset + step) % len(negatives)], "relation_train")
                )
    test = []
    vocabulary = [word for group in groups for word in group]
    rng = random.Random(seed)
    for left, right in HELD_RELATIONS:
        same_group = next(words for words in groups if left in words)
        negatives = [word for word in vocabulary if word not in same_group]
        rng.shuffle(negatives)
        for negative in negatives[:4]:
            test.append(Triplet(left, right, negative, "relation_disjoint"))
    return training, test


def build_context_data(seed=20260928):
    rng = random.Random(seed)
    contexts = {}
    for category, group in CONTEXT_GROUPS.items():
        contexts[category] = {
            word: [pattern.format(word=word) for pattern in group["patterns"]]
            for word in (*group["known"], *group["new"])
        }
    training = []
    categories = list(CONTEXT_GROUPS)
    for category, group in CONTEXT_GROUPS.items():
        other_categories = [name for name in categories if name != category]
        for new_word in group["new"]:
            for new_context in contexts[category][new_word]:
                for known_word in group["known"]:
                    positive = rng.choice(contexts[category][known_word])
                    negative_category = rng.choice(other_categories)
                    negative_word = rng.choice(CONTEXT_GROUPS[negative_category]["known"])
                    negative = rng.choice(contexts[negative_category][negative_word])
                    training.append(
                        Triplet(new_context, positive, negative, "context_exposure")
                    )
    test = []
    for category, group in CONTEXT_GROUPS.items():
        other_known = [
            word
            for other_category, other_group in CONTEXT_GROUPS.items()
            if other_category != category
            for word in other_group["known"]
        ]
        for new_word in group["new"]:
            for positive in group["known"]:
                for negative in other_known[:3]:
                    test.append(
                        Triplet(new_word, positive, negative, "lexical_context_transfer")
                    )
    return training, test


def matched_no_context_training(base_training, target_length):
    return list(itertools.islice(itertools.cycle(base_training), target_length))


def encode_triplets(model, encoder, triplets):
    texts = []
    for item in triplets:
        texts.extend([item.anchor, item.positive, item.negative])
    output = model(encoder.encode(texts))
    return split_spikes(output)


@torch.no_grad()
def evaluate(model, encoder, triplets, base):
    a, p, n = encode_triplets(model, encoder, triplets)
    taus, weights = base["distance"]["taus"], base["distance"]["weights"]
    return {
        "temporal": distance_statistics(a, p, n, taus, weights),
        "population": distance_statistics(a, p, n, rate_only=True),
    }


def aggregate(results, samples):
    summary = {}
    for model_name in ("snn", "ann"):
        summary[model_name] = {}
        for condition in ("no_context", "context_exposed"):
            selected = [
                row for row in results if row["model"] == model_name and row["condition"] == condition
            ]
            summary[model_name][condition] = {}
            for test_name in ("relation_disjoint", "lexical_context_transfer"):
                summary[model_name][condition][test_name] = {}
                for metric in ("temporal", "population"):
                    summary[model_name][condition][test_name][metric] = {}
                    for field in ("sta", "positive_distance", "negative_distance", "margin"):
                        values = [row["evaluation"][test_name][metric][field] for row in selected]
                        summary[model_name][condition][test_name][metric][field] = bootstrap_interval(
                            values, samples
                        )
    summary["paired_context_gain"] = {}
    for model_name in ("snn", "ann"):
        summary["paired_context_gain"][model_name] = {}
        no_context = {
            row["seed"]: row for row in results if row["model"] == model_name and row["condition"] == "no_context"
        }
        exposed = {
            row["seed"]: row for row in results if row["model"] == model_name and row["condition"] == "context_exposed"
        }
        for metric in ("temporal", "population"):
            values = [
                exposed[seed]["evaluation"]["lexical_context_transfer"][metric]["sta"]
                - no_context[seed]["evaluation"]["lexical_context_transfer"][metric]["sta"]
                for seed in sorted(no_context)
            ]
            summary["paired_context_gain"][model_name][metric] = bootstrap_interval(values, samples)
    return summary


def run(config_path="configs/generalization.toml", epochs_override=None, seeds_override=None):
    with Path(config_path).open("rb") as handle:
        config = tomllib.load(handle)
    _, base = load_configs(Path(config["e11_config"]))
    with Path(config["e11_config"]).open("rb") as handle:
        e11 = tomllib.load(handle)
    seeds = seeds_override or e11["seeds"]
    epochs = epochs_override or base["run"]["epochs"]
    encoder = EventEncoder(**base["input"])
    relation_training, relation_test = build_relation_data()
    context_training, lexical_test = build_context_data()
    exposed_training = relation_training + context_training
    control_training = matched_no_context_training(relation_training, len(exposed_training))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for model_name in config["models"]:
        for condition in config["conditions"]:
            training = exposed_training if condition == "context_exposed" else control_training
            for seed in seeds:
                if model_name == "snn":
                    model, loss = train_snn(seed, base, encoder, training, epochs)
                else:
                    model, loss = train_ann(seed, base, encoder, training, epochs)
                evaluation = {
                    "relation_disjoint": evaluate(model, encoder, relation_test, base),
                    "lexical_context_transfer": evaluate(model, encoder, lexical_test, base),
                }
                result = {
                    "model": model_name,
                    "condition": condition,
                    "seed": seed,
                    "loss": loss,
                    "train_examples": len(training),
                    "evaluation": evaluation,
                }
                results.append(result)
                run_dir = output_dir / model_name / condition
                run_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), run_dir / f"seed_{seed}.pt")
                (run_dir / f"seed_{seed}.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(
                    f"model={model_name} condition={condition} seed={seed} "
                    f"relation={evaluation['relation_disjoint']['population']['sta']:.3f} "
                    f"lexical={evaluation['lexical_context_transfer']['population']['sta']:.3f}"
                )
    summary = {
        "experiment": "E1-generalization",
        "protocols": {
            "relation_disjoint": (
                "Both lexemes appear in training, but the tested positive edge and its reverse are excluded."
            ),
            "lexical_context_transfer": (
                "New lexemes occur only inside supervised semantic context triplets during training, "
                "then are evaluated as isolated strings. This is not unsupervised exposure."
            ),
            "no_context_control": (
                "Same number of training examples/updates as context_exposed, using cycled relation data."
            ),
        },
        "held_relations": HELD_RELATIONS,
        "new_lexemes": {
            category: group["new"] for category, group in CONTEXT_GROUPS.items()
        },
        "seeds": seeds,
        "epochs": epochs,
        "relation_train_examples": len(relation_training),
        "context_examples": len(context_training),
        "relation_test_examples": len(relation_test),
        "lexical_test_examples": len(lexical_test),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate(results, config["bootstrap_samples"]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generalizacion relation-disjoint y contextual")
    parser.add_argument("--config", default="configs/generalization.toml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    run(args.config, args.epochs, args.seeds)


if __name__ == "__main__":
    main()

