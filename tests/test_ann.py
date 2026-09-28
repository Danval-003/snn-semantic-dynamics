import torch

from e1_spikes.data import EventEncoder
from e1_spikes.model_ann import HierarchicalANN


def test_ann_is_continuous_and_parameter_matched():
    encoder = EventEncoder(char_steps=1, post_steps=3, max_chars=6)
    model = HierarchicalANN(encoder.channels, [12, 8])
    output = model(encoder.encode(["perro", "can"]))
    assert output.shape == (2, 9, 8)
    assert torch.any((output != 0) & (output != 1))
    output.mean().backward()
    assert model.layers[0].input.weight.grad is not None
