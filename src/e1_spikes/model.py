from __future__ import annotations

import math

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    """Heaviside en forward y derivada rapida-sigmoide en backward."""

    @staticmethod
    def forward(ctx, membrane_minus_threshold: torch.Tensor):
        ctx.save_for_backward(membrane_minus_threshold)
        return (membrane_minus_threshold >= 0).to(membrane_minus_threshold.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        scale = 5.0
        surrogate = scale / (1.0 + scale * x.abs()).pow(2)
        return grad_output * surrogate


spike_fn = SurrogateSpike.apply


class RecurrentALIFLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        size: int,
        threshold: float,
        recurrent: bool,
        adaptive: bool,
        heterogeneous: bool,
        layer_index: int,
        tau_mode: str | None = None,
    ):
        super().__init__()
        self.size = size
        self.threshold = threshold
        self.recurrent = recurrent
        self.adaptive = adaptive
        self.input = nn.Linear(input_size, size, bias=True)
        self.rec = nn.Linear(size, size, bias=False) if recurrent else None

        # Capas posteriores comienzan con memoria mas lenta. Cada neurona puede
        # aprender su propia constante temporal.
        if tau_mode is None:
            tau_mode = "heterogeneous_learnable" if heterogeneous else "layer_learnable"
        valid_modes = {
            "single_fixed",
            "layer_fixed",
            "heterogeneous_fixed",
            "heterogeneous_learnable",
            "layer_learnable",
        }
        if tau_mode not in valid_modes:
            raise ValueError(f"tau_mode desconocido: {tau_mode}")
        base_beta = 0.68 if tau_mode == "single_fixed" else min(0.55 + 0.13 * layer_index, 0.92)
        is_heterogeneous = tau_mode in {"heterogeneous_fixed", "heterogeneous_learnable"}
        if is_heterogeneous:
            initial = torch.linspace(base_beta - 0.08, base_beta + 0.08, size)
        else:
            initial = torch.full((size,), base_beta)
        initial = initial.clamp(0.1, 0.97)
        beta_logit = torch.logit(initial)
        if tau_mode in {"single_fixed", "layer_fixed", "heterogeneous_fixed"}:
            self.register_buffer("beta_logit", beta_logit)
        else:
            self.beta_logit = nn.Parameter(beta_logit)
        self.tau_mode = tau_mode
        self.adapt_logit = nn.Parameter(torch.full((size,), math.log(0.9 / 0.1)))
        self.adapt_strength = nn.Parameter(torch.full((size,), 0.12))
        # Un evento grafemico aislado debe poder cruzar el umbral en parte de
        # la primera poblacion; de otro modo solo se aprende desde biases.
        nn.init.xavier_uniform_(self.input.weight, gain=2.5)
        nn.init.zeros_(self.input.bias)
        if self.rec is not None:
            nn.init.orthogonal_(self.rec.weight, gain=0.25)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = inputs.shape
        membrane = inputs.new_zeros(batch, self.size)
        previous_spike = inputs.new_zeros(batch, self.size)
        adaptation = inputs.new_zeros(batch, self.size)
        beta = torch.sigmoid(self.beta_logit).clamp(0.05, 0.995)
        adapt_decay = torch.sigmoid(self.adapt_logit).clamp(0.5, 0.999)
        outputs = []
        for time_index in range(steps):
            current = self.input(inputs[:, time_index])
            if self.rec is not None:
                current = current + self.rec(previous_spike)
            if self.adaptive:
                adaptation = adapt_decay * adaptation + previous_spike
                dynamic_threshold = self.threshold + torch.relu(self.adapt_strength) * adaptation
            else:
                dynamic_threshold = self.threshold
            membrane = beta * membrane + current - previous_spike * self.threshold
            previous_spike = spike_fn(membrane - dynamic_threshold)
            outputs.append(previous_spike)
        return torch.stack(outputs, dim=1)


class HierarchicalSNN(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_sizes: list[int],
        threshold: float = 0.55,
        recurrent: bool = True,
        adaptive: bool = True,
        heterogeneous: bool = True,
        tau_mode: str | None = None,
    ):
        super().__init__()
        sizes = [input_channels, *hidden_sizes]
        self.layers = nn.ModuleList(
            RecurrentALIFLayer(
                sizes[i],
                sizes[i + 1],
                threshold,
                recurrent,
                adaptive,
                heterogeneous,
                i,
                tau_mode,
            )
            for i in range(len(hidden_sizes))
        )

    def forward(self, events: torch.Tensor, return_all: bool = False):
        activity = events
        all_layers = []
        for layer in self.layers:
            activity = layer(activity)
            all_layers.append(activity)
        return all_layers if return_all else activity
