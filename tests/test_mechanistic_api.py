import torch

from e1_spikes.data import EventEncoder
from e1_spikes.models import make_model
from e1_spikes.readouts import extract_readouts, trajectory_mask


MODEL = {
    "hidden_sizes": [8, 6, 4],
    "threshold": 0.35,
    "recurrent": True,
    "adaptive": True,
    "heterogeneous": True,
}


def test_explicit_legacy_encoding_is_bit_exact():
    encoder = EventEncoder(char_steps=2, post_steps=12, max_chars=24)
    texts = ["perro", "automóvil", "se detuvo"]
    legacy = encoder.encode(texts)
    explicit = encoder.encode_explicit(
        texts, settling_steps=encoder.post_steps, minimum_steps=encoder.total_steps
    )
    assert torch.equal(legacy, explicit.events)
    assert explicit.stimulus_ends.tolist() == [8, 16, 16]
    assert explicit.mode == "legacy"


def test_natural_forward_is_bit_exact_for_snn_and_ann():
    encoder = EventEncoder(char_steps=2, post_steps=12, max_chars=24)
    encoded = encoder.encode_explicit(
        ["perro", "automóvil"], minimum_steps=encoder.total_steps
    )
    for kind in ("snn", "ann"):
        torch.manual_seed(17)
        model = make_model(kind, encoder.channels, MODEL)
        legacy = model(encoded.events, return_all=True)
        explicit = model(
            encoded.events,
            return_all=True,
            stimulus_ends=encoded.stimulus_ends,
            settling_intervention="natural",
        )
        assert all(torch.equal(left, right) for left, right in zip(legacy, explicit, strict=True))


def test_interventions_never_change_stimulus_activity():
    encoder = EventEncoder(char_steps=2, post_steps=12, max_chars=24)
    encoded = encoder.encode_explicit(["can", "automóvil"], settling_steps=16)
    for kind in ("snn", "ann"):
        torch.manual_seed(23)
        model = make_model(kind, encoder.channels, MODEL)
        natural = model(encoded.events)
        for intervention in ("no_recurrent", "intrinsic_only", "reset_state"):
            changed = model(
                encoded.events,
                stimulus_ends=encoded.stimulus_ends,
                settling_intervention=intervention,
            )
            for row, end in enumerate(encoded.stimulus_ends.tolist()):
                assert torch.equal(natural[row, : end + 1], changed[row, : end + 1])


def test_readouts_are_relative_and_non_interchangeable():
    activity = torch.arange(10).reshape(1, 10, 1).float()
    views = extract_readouts(activity, [3], settling_steps=3, local_width=2)
    assert views["stimulus_only"].item() == sum(range(4))
    assert views["instantaneous"].item() == 6
    assert views["local_window"].item() == 5 + 6
    assert views["post_only"].item() == 4 + 5 + 6
    assert views["full_trajectory"].item() == sum(range(7))
    masked = trajectory_mask(torch.ones(2, 10, 1), torch.tensor([2, 5]), 3)
    assert masked[0].sum().item() == 6
    assert masked[1].sum().item() == 9
