from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .distance import multiscale_distance, rate_distance


@dataclass
class Metrics:
    temporal_sta: float
    rate_sta: float
    temporal_advantage: float
    spike_rate: float
    active_neurons: float
    dead_neurons: float
    mean_positive_distance: float
    mean_negative_distance: float

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def triplet_metrics(a, p, n, taus, weights) -> Metrics:
    d_ap = multiscale_distance(a, p, taus, weights)
    d_an = multiscale_distance(a, n, taus, weights)
    r_ap = rate_distance(a, p)
    r_an = rate_distance(a, n)
    joined = torch.cat([a, p, n], dim=0)
    per_neuron = joined.mean(dim=(0, 1))
    temporal_sta = (d_ap < d_an).float().mean().item()
    rate_sta = (r_ap < r_an).float().mean().item()
    return Metrics(
        temporal_sta=temporal_sta,
        rate_sta=rate_sta,
        temporal_advantage=temporal_sta - rate_sta,
        spike_rate=joined.mean().item(),
        active_neurons=(per_neuron > 0).float().mean().item(),
        dead_neurons=(per_neuron < 1e-5).float().mean().item(),
        mean_positive_distance=d_ap.mean().item(),
        mean_negative_distance=d_an.mean().item(),
    )

