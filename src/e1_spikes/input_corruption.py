from __future__ import annotations

import torch


def _occupied(events: torch.Tensor, batch_index: int, stimulus_steps: int):
    return torch.nonzero(
        events[batch_index, :stimulus_steps].sum(dim=-1) > 0, as_tuple=False
    ).flatten()


def reverse_graphemes(events: torch.Tensor, stimulus_steps: int) -> torch.Tensor:
    """Invierte la secuencia de grafemas conservando sus tiempos regulares."""
    result = events.clone()
    for batch_index in range(events.shape[0]):
        positions = _occupied(events, batch_index, stimulus_steps)
        result[batch_index, positions] = events[batch_index, positions].flip(0)
    return result


def permute_graphemes(
    events: torch.Tensor, stimulus_steps: int, generator: torch.Generator
) -> torch.Tensor:
    """Permuta identidades en los tiempos ocupados; conserva ritmo y longitud."""
    result = events.clone()
    for batch_index in range(events.shape[0]):
        positions = _occupied(events, batch_index, stimulus_steps)
        order = torch.randperm(len(positions), generator=generator, device=events.device)
        result[batch_index, positions] = events[batch_index, positions[order]]
    return result


def ordered_timing_jitter(
    events: torch.Tensor,
    stimulus_steps: int,
    generator: torch.Generator,
    max_gap: int = 3,
) -> torch.Tensor:
    """Conserva orden y comienzo, variando intervalos entre 1 y `max_gap`."""
    if max_gap < 1:
        raise ValueError("max_gap debe ser al menos 1")
    result = torch.zeros_like(events)
    # La ventana post-estimulo ya es cero por construccion, pero copiarla hace
    # explicita la invariancia si posteriormente se agregan canales de control.
    result[:, stimulus_steps:] = events[:, stimulus_steps:]
    for batch_index in range(events.shape[0]):
        positions = _occupied(events, batch_index, stimulus_steps)
        if len(positions) == 0:
            continue
        new_positions = [0]
        for token_index in range(1, len(positions)):
            remaining = len(positions) - token_index - 1
            maximum_that_fits = stimulus_steps - 1 - new_positions[-1] - remaining
            upper = min(max_gap, maximum_that_fits)
            gap = int(
                torch.randint(
                    1, upper + 1, (1,), generator=generator, device=events.device
                ).item()
            )
            new_positions.append(new_positions[-1] + gap)
        new_positions = torch.tensor(new_positions, device=events.device)
        result[batch_index, new_positions] = events[batch_index, positions]
    return result


def bag_of_graphemes(
    events: torch.Tensor, stimulus_steps: int, generator: torch.Generator
) -> torch.Tensor:
    """Conserva el multiconjunto de grafemas, destruyendo orden y ritmo."""
    result = torch.zeros_like(events)
    result[:, stimulus_steps:] = events[:, stimulus_steps:]
    for batch_index in range(events.shape[0]):
        positions = _occupied(events, batch_index, stimulus_steps)
        new_positions = torch.randperm(
            stimulus_steps, generator=generator, device=events.device
        )[: len(positions)].sort().values
        token_order = torch.randperm(
            len(positions), generator=generator, device=events.device
        )
        result[batch_index, new_positions] = events[batch_index, positions[token_order]]
    return result


def assert_preserves_channel_counts(original: torch.Tensor, corrupted: torch.Tensor):
    if not torch.equal(original.sum(dim=1), corrupted.sum(dim=1)):
        raise AssertionError("La corrupcion altero el conteo de eventos por canal")
