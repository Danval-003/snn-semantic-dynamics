from __future__ import annotations

import math

import torch


def exp_filter(spikes: torch.Tensor, tau: float) -> torch.Tensor:
    """Filtro causal exponencial diferenciable para tensores [B,T,N]."""
    decay = math.exp(-1.0 / tau)
    state = torch.zeros_like(spikes[:, 0])
    filtered = []
    for time_index in range(spikes.shape[1]):
        state = decay * state + spikes[:, time_index]
        filtered.append(state)
    return torch.stack(filtered, dim=1)


def van_rossum_distance(a: torch.Tensor, b: torch.Tensor, tau: float) -> torch.Tensor:
    difference = exp_filter(a, tau) - exp_filter(b, tau)
    return difference.square().mean(dim=(1, 2))


def multiscale_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    taus: list[float],
    weights: list[float],
) -> torch.Tensor:
    if len(taus) != len(weights) or not taus:
        raise ValueError("taus y weights deben tener la misma longitud no vacia")
    total_weight = sum(weights)
    return sum(
        weight * van_rossum_distance(a, b, tau)
        for tau, weight in zip(taus, weights, strict=True)
    ) / total_weight


def rate_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.sum(dim=1) - b.sum(dim=1)).square().mean(dim=1)

