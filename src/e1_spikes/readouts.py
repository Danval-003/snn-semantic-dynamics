from __future__ import annotations

import torch


READOUTS = (
    "stimulus_only",
    "instantaneous",
    "local_window",
    "post_only",
    "full_trajectory",
)


def trajectory_mask(activity: torch.Tensor, stimulus_ends: torch.Tensor, settling_steps: int):
    """Mask padding after each example's explicit relative endpoint."""
    times = torch.arange(activity.shape[1], device=activity.device).unsqueeze(0)
    endpoints = stimulus_ends.to(activity.device).unsqueeze(1) + settling_steps
    return activity * (times <= endpoints).unsqueeze(2)


def extract_readouts(
    activity: torch.Tensor,
    stimulus_ends,
    settling_steps: int,
    local_width: int = 4,
):
    """Extract all non-interchangeable E1.5 population representations."""
    if settling_steps < 0:
        raise ValueError("settling_steps debe ser no negativo")
    if local_width < 1:
        raise ValueError("local_width debe ser positivo")
    ends = stimulus_ends.tolist() if isinstance(stimulus_ends, torch.Tensor) else stimulus_ends
    result = {name: [] for name in READOUTS}
    for row, raw_end in enumerate(ends):
        end = int(raw_end)
        endpoint = end + settling_steps
        if endpoint >= activity.shape[1]:
            raise ValueError("La actividad no contiene el settling explícito completo")
        result["stimulus_only"].append(activity[row, : end + 1].sum(0))
        result["instantaneous"].append(activity[row, endpoint])
        local_start = max(end + 1, endpoint - local_width + 1)
        result["local_window"].append(activity[row, local_start : endpoint + 1].sum(0))
        result["post_only"].append(activity[row, end + 1 : endpoint + 1].sum(0))
        result["full_trajectory"].append(activity[row, : endpoint + 1].sum(0))
    return {name: torch.stack(values) for name, values in result.items()}
