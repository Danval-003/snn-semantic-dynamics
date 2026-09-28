from e1_spikes.factorial import audit, levenshtein, orthographic_similarity


def test_levenshtein_and_similarity():
    assert levenshtein("casa", "caso") == 1
    assert orthographic_similarity("casa", "caso") == 0.75


def test_factorial_audit_requires_balanced_cells():
    base = {
        "left": "a",
        "right": "b",
        "relation": "test",
        "pos": "noun",
        "exposure": "test",
    }
    pairs = []
    for orthography in ("similar", "different"):
        for semantics in ("similar", "different"):
            pair = {**base, "left": f"{orthography}{semantics}", "orthography": orthography, "semantics": semantics}
            pairs.append(pair)
    _, report = audit(pairs)
    assert set(report["counts"].values()) == {1}
