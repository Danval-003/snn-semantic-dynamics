from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import torch
import torch.nn.functional as functional

from .data import EventEncoder, batch_triplets, make_smoke_triplets, split_iid
from .distance import multiscale_distance
from .metrics import triplet_metrics
from .model import HierarchicalSNN


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def encode_triplet_batch(model, encoder, triplets, device):
    texts = []
    for item in triplets:
        texts.extend([item.anchor, item.positive, item.negative])
    spikes = model(encoder.encode(texts).to(device))
    batch = len(triplets)
    spikes = spikes.reshape(batch, 3, spikes.shape[1], spikes.shape[2])
    return spikes[:, 0], spikes[:, 1], spikes[:, 2]


def activity_regularizer(spikes, target_rate: float, balance_weight: float):
    population_rate = spikes.mean()
    per_neuron_rate = spikes.mean(dim=(0, 1))
    rate_loss = (population_rate - target_rate).square()
    balance_loss = per_neuron_rate.var()
    return rate_loss + balance_weight * balance_loss


@torch.no_grad()
def evaluate(model, encoder, triplets, device, taus, weights):
    model.eval()
    a, p, n = encode_triplet_batch(model, encoder, triplets, device)
    return triplet_metrics(a, p, n, taus, weights)


def run(config_path: str | Path, epochs_override: int | None = None):
    config_path = Path(config_path)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    run_config = config["run"]
    seed = run_config["seed"]
    seed_everything(seed)
    device = resolve_device(run_config["device"])
    encoder = EventEncoder(**config["input"])
    model = HierarchicalSNN(encoder.channels, **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config["learning_rate"])
    all_triplets = make_smoke_triplets(seed)
    train_triplets, validation_triplets = split_iid(all_triplets)
    taus = config["distance"]["taus"]
    weights = config["distance"]["weights"]
    epochs = epochs_override if epochs_override is not None else run_config["epochs"]
    rng = random.Random(seed + 1)

    initial = evaluate(model, encoder, validation_triplets, device, taus, weights)
    history = [{"epoch": 0, **initial.to_dict()}]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in batch_triplets(train_triplets, run_config["batch_size"], rng):
            a, p, n = encode_triplet_batch(model, encoder, batch, device)
            d_ap = multiscale_distance(a, p, taus, weights)
            d_an = multiscale_distance(a, n, taus, weights)
            semantic_loss = functional.relu(
                d_ap - d_an + run_config["margin"]
            ).mean()
            joined = torch.cat([a, p, n], dim=0)
            regularization = activity_regularizer(
                joined,
                config["regularization"]["target_rate"],
                config["regularization"]["balance_weight"],
            )
            loss = semantic_loss + config["regularization"]["rate_weight"] * regularization
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, encoder, validation_triplets, device, taus, weights)
        train_metrics = evaluate(model, encoder, train_triplets, device, taus, weights)
        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "train_temporal_sta": train_metrics.temporal_sta,
            **metrics.to_dict(),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} loss={row['loss']:.4f} "
            f"STA_train={train_metrics.temporal_sta:.3f} "
            f"STA_val={metrics.temporal_sta:.3f} STA_r={metrics.rate_sta:.3f} "
            f"rate={metrics.spike_rate:.3f} active={metrics.active_neurons:.3f}"
        )

    output_dir = Path(run_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "E1-smoke-v0",
        "scope": "pipeline/IID; no demuestra generalizacion lexical-disjoint",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_triplets": len(train_triplets),
        "validation_triplets": len(validation_triplets),
        "elapsed_seconds": time.perf_counter() - started,
        "initial": initial.to_dict(),
        "final": history[-1],
    }
    final = history[-1]
    summary["gates"] = {
        "G1_microfit": (
            final["temporal_sta"] >= 0.70
            and final["temporal_sta"] - initial.temporal_sta >= 0.15
            and 0.01 <= final["spike_rate"] <= 0.25
            and final["active_neurons"] >= 0.50
        ),
        "H3_timing_supported": None,
        "H3_note": (
            "No se decide con una semilla. Comparar varias semillas y control "
            "de barajado temporal; temporal_advantage actual es solo diagnostico."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in history), encoding="utf-8"
    )
    torch.save({"model": model.state_dict(), "config": config}, output_dir / "model.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ejecuta el smoke test de E1")
    parser.add_argument("--config", default="configs/smoke.toml")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    run(args.config, args.epochs)


if __name__ == "__main__":
    main()
