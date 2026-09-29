from pathlib import Path

from e1_spikes.factorial import audit, load_pairs
from e1_spikes.lexical_holdout import build_lexical_holdout


def test_lexical_holdout_is_balanced_and_separate_from_primary_factorial():
    holdout = build_lexical_holdout()
    _, report = audit(holdout)
    assert report["counts"] == {
        "orth+_semantic+": 12,
        "orth+_semantic-": 12,
        "orth-_semantic+": 12,
        "orth-_semantic-": 12,
    }
    primary = load_pairs(Path("data/factorial_pairs.jsonl"))
    primary_pairs = {frozenset((row["left"], row["right"])) for row in primary}
    holdout_pairs = {frozenset((row["left"], row["right"])) for row in holdout}
    assert primary_pairs.isdisjoint(holdout_pairs)
    assert all(row["exposure"] == "evaluation_only" for row in holdout)
