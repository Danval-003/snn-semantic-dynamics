import torch

from e1_spikes.e12a import distance_statistics


def test_hierarchy_margin_is_negative_minus_positive():
    anchor = torch.zeros(2, 8, 3)
    positive = anchor.clone()
    negative = anchor.clone()
    negative[:, 3, 1] = 1
    result = distance_statistics(anchor, positive, negative, [2.0], [1.0])
    assert result["positive_distance"] == 0
    assert result["negative_distance"] > 0
    assert result["margin"] > 0
    assert result["sta"] == 1
