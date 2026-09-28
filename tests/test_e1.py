import torch

from e1_spikes.data import EventEncoder
from e1_spikes.corruption import global_time_shuffle, local_time_shuffle
from e1_spikes.distance import multiscale_distance, rate_distance, van_rossum_distance
from e1_spikes.model import HierarchicalSNN


def test_encoder_preserves_accents_and_has_post_window():
    encoder = EventEncoder(char_steps=2, post_steps=5, max_chars=4)
    events = encoder.encode(["ñá!"])
    assert events.shape == (1, 13, encoder.channels)
    assert events[0, 0].sum() == 1
    assert events[0, 2].sum() == 1
    assert events[0, 4].sum() == 1
    assert events[0, -5:].sum() == 0


def test_timing_distance_detects_equal_rate_different_timing():
    first = torch.zeros(1, 8, 2)
    second = torch.zeros_like(first)
    first[0, 1, 0] = 1
    second[0, 5, 0] = 1
    assert rate_distance(first, second).item() == 0
    assert van_rossum_distance(first, second, tau=2).item() > 0


def test_multiscale_distance_is_zero_for_identical_activity():
    spikes = torch.randint(0, 2, (3, 10, 4)).float()
    distance = multiscale_distance(spikes, spikes, [2, 5], [0.5, 0.5])
    assert torch.allclose(distance, torch.zeros(3))


def test_snn_forward_is_binary_and_differentiable():
    encoder = EventEncoder(char_steps=1, post_steps=3, max_chars=6)
    model = HierarchicalSNN(encoder.channels, [12, 8])
    output = model(encoder.encode(["perro", "can"]))
    assert output.shape == (2, 9, 8)
    assert set(output.detach().unique().tolist()).issubset({0.0, 1.0})
    output.mean().backward()
    assert model.layers[0].input.weight.grad is not None


def test_temporal_corruptions_preserve_each_neuron_count():
    spikes = torch.randint(0, 2, (4, 15, 7)).float()
    original_counts = spikes.sum(dim=1)
    local = local_time_shuffle(spikes, 3, torch.Generator().manual_seed(1))
    global_ = global_time_shuffle(spikes, torch.Generator().manual_seed(2))
    assert torch.equal(local.sum(dim=1), original_counts)
    assert torch.equal(global_.sum(dim=1), original_counts)
    assert not torch.equal(global_, spikes)


def test_neuronal_timescale_modes_fix_or_learn_beta_as_requested():
    fixed = HierarchicalSNN(10, [6, 6, 6], tau_mode="single_fixed")
    learned = HierarchicalSNN(10, [6, 6, 6], tau_mode="heterogeneous_learnable")
    assert all(not isinstance(layer.beta_logit, torch.nn.Parameter) for layer in fixed.layers)
    assert all(isinstance(layer.beta_logit, torch.nn.Parameter) for layer in learned.layers)
    fixed_means = [torch.sigmoid(layer.beta_logit).mean().item() for layer in fixed.layers]
    assert max(fixed_means) - min(fixed_means) < 1e-6
