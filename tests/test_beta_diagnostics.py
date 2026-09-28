import numpy as np

from e1_spikes.beta_diagnostics import correlation, rankdata


def test_rankdata_uses_average_rank_for_ties():
    ranks = rankdata(np.array([2.0, 1.0, 2.0]))
    assert np.allclose(ranks, [1.5, 0.0, 1.5])


def test_correlation_handles_constant_vector():
    assert correlation([1, 1, 1], [1, 2, 3]) is None
