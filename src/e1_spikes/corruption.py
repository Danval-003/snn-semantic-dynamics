from __future__ import annotations

import torch


def global_time_shuffle(spikes: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permutacion temporal independiente para cada muestra y neurona."""
    by_neuron = spikes.transpose(1, 2)
    noise = torch.rand(
        by_neuron.shape,
        generator=generator,
        device=by_neuron.device,
        dtype=by_neuron.dtype,
    )
    permutation = noise.argsort(dim=-1)
    return by_neuron.gather(-1, permutation).transpose(1, 2)


def local_time_shuffle(
    spikes: torch.Tensor, radius: int, generator: torch.Generator
) -> torch.Tensor:
    """Corrupcion local mediante `radius` rondas de swaps adyacentes.

    Cada ronda mueve un spike como maximo un paso. Los swaps preservan tanto la
    identidad neuronal como el conteo exacto, incluso cuando hay spikes vecinos.
    """
    if radius < 0:
        raise ValueError("radius debe ser no negativo")
    shuffled = spikes.transpose(1, 2).clone()
    steps = shuffled.shape[-1]
    for round_index in range(radius):
        start = round_index % 2
        left = torch.arange(start, steps - 1, 2, device=shuffled.device)
        if left.numel() == 0:
            continue
        right = left + 1
        swap = torch.rand(
            (*shuffled.shape[:-1], left.numel()),
            generator=generator,
            device=shuffled.device,
        ) < 0.5
        left_values = shuffled[..., left].clone()
        right_values = shuffled[..., right].clone()
        shuffled[..., left] = torch.where(swap, right_values, left_values)
        shuffled[..., right] = torch.where(swap, left_values, right_values)
    return shuffled.transpose(1, 2)


def assert_preserves_per_neuron_counts(original: torch.Tensor, corrupted: torch.Tensor):
    if not torch.equal(original.sum(dim=1), corrupted.sum(dim=1)):
        raise AssertionError("La corrupcion temporal altero el conteo de spikes")
