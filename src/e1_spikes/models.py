from __future__ import annotations

from .model import HierarchicalSNN
from .model_ann import HierarchicalANN


MODEL_KINDS = ("snn", "ann")


def make_model(kind: str, input_channels: int, model_config: dict):
    """Construct paired recurrent models behind one analysis interface."""
    if kind == "snn":
        return HierarchicalSNN(input_channels, **model_config)
    if kind == "ann":
        return HierarchicalANN(input_channels, **model_config)
    raise ValueError(f"Modelo desconocido: {kind}; esperado uno de {MODEL_KINDS}")
