from __future__ import annotations

import math

import torch
from torch import nn


class AdaptiveLeakyRNNLayer(nn.Module):
    """Contraparte continua de una capa ALIF, con igual parametrizacion."""

    def __init__(self, input_size, size, recurrent, adaptive, layer_index):
        super().__init__()
        self.size = size
        self.recurrent = recurrent
        self.adaptive = adaptive
        self.input = nn.Linear(input_size, size, bias=True)
        self.rec = nn.Linear(size, size, bias=False) if recurrent else None
        base_beta = min(0.55 + 0.13 * layer_index, 0.92)
        initial = torch.linspace(base_beta - 0.08, base_beta + 0.08, size).clamp(0.1, 0.97)
        self.beta_logit = nn.Parameter(torch.logit(initial))
        self.adapt_logit = nn.Parameter(torch.full((size,), math.log(0.9 / 0.1)))
        self.adapt_strength = nn.Parameter(torch.full((size,), 0.12))
        nn.init.xavier_uniform_(self.input.weight, gain=2.5)
        nn.init.zeros_(self.input.bias)
        if self.rec is not None:
            nn.init.orthogonal_(self.rec.weight, gain=0.25)

    def forward(self, inputs):
        batch, steps, _ = inputs.shape
        state = inputs.new_zeros(batch, self.size)
        adaptation = inputs.new_zeros(batch, self.size)
        beta = torch.sigmoid(self.beta_logit).clamp(0.05, 0.995)
        adapt_decay = torch.sigmoid(self.adapt_logit).clamp(0.5, 0.999)
        outputs = []
        for time_index in range(steps):
            current = self.input(inputs[:, time_index])
            if self.rec is not None:
                current = current + self.rec(state)
            if self.adaptive:
                adaptation = adapt_decay * adaptation + state.abs()
                current = current - torch.relu(self.adapt_strength) * adaptation
            candidate = torch.tanh(current)
            state = beta * state + (1.0 - beta) * candidate
            outputs.append(state)
        return torch.stack(outputs, dim=1)


class HierarchicalANN(nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_sizes,
        threshold=0.35,
        recurrent=True,
        adaptive=True,
        heterogeneous=True,
    ):
        super().__init__()
        # threshold/heterogeneous se aceptan para compartir exactamente el
        # diccionario de configuracion; la ANN es continua y heterogenea.
        del threshold, heterogeneous
        sizes = [input_channels, *hidden_sizes]
        self.layers = nn.ModuleList(
            AdaptiveLeakyRNNLayer(
                sizes[index], sizes[index + 1], recurrent, adaptive, index
            )
            for index in range(len(hidden_sizes))
        )

    def forward(self, events, return_all=False):
        activity = events
        layers = []
        for layer in self.layers:
            activity = layer(activity)
            layers.append(activity)
        return layers if return_all else activity

