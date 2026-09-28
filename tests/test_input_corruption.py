import torch

from e1_spikes.data import EventEncoder
from e1_spikes.input_corruption import (
    bag_of_graphemes,
    ordered_timing_jitter,
    permute_graphemes,
    reverse_graphemes,
)


def event_sequence(events):
    occupied = events[0].sum(dim=-1) > 0
    return events[0, occupied].argmax(dim=-1).tolist()


def test_input_corruptions_preserve_channel_counts_and_post_window():
    encoder = EventEncoder(char_steps=2, post_steps=5, max_chars=8)
    original = encoder.encode(["casa"])
    stimulus_steps = encoder.max_chars * encoder.char_steps
    corruptions = [
        reverse_graphemes(original, stimulus_steps),
        permute_graphemes(original, stimulus_steps, torch.Generator().manual_seed(1)),
        ordered_timing_jitter(original, stimulus_steps, torch.Generator().manual_seed(2)),
        bag_of_graphemes(original, stimulus_steps, torch.Generator().manual_seed(3)),
    ]
    for corrupted in corruptions:
        assert torch.equal(corrupted.sum(dim=1), original.sum(dim=1))
        assert corrupted[:, stimulus_steps:].sum() == 0


def test_reverse_and_ordered_jitter_have_expected_order():
    encoder = EventEncoder(char_steps=2, post_steps=3, max_chars=8)
    original = encoder.encode(["casa"])
    stimulus_steps = encoder.max_chars * encoder.char_steps
    original_sequence = event_sequence(original)
    reversed_events = reverse_graphemes(original, stimulus_steps)
    jittered = ordered_timing_jitter(
        original, stimulus_steps, torch.Generator().manual_seed(4)
    )
    assert event_sequence(reversed_events) == list(reversed(original_sequence))
    assert event_sequence(jittered) == original_sequence
    occupied = torch.nonzero(jittered[0].sum(dim=-1) > 0).flatten()
    assert occupied[0].item() == 0
    assert set((occupied[1:] - occupied[:-1]).tolist()).issubset({1, 2, 3})
