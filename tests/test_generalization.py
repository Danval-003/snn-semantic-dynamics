from e1_spikes.generalization import (
    CONTEXT_GROUPS,
    HELD_RELATIONS,
    build_context_data,
    build_relation_data,
    unordered_pair,
)


def test_held_relations_are_absent_from_training():
    training, test = build_relation_data()
    held = {unordered_pair(*pair) for pair in HELD_RELATIONS}
    assert all(unordered_pair(item.anchor, item.positive) not in held for item in training)
    assert {unordered_pair(item.anchor, item.positive) for item in test} == held


def test_new_lexemes_only_appear_inside_contexts_during_training():
    training, test = build_context_data()
    new_words = {word for group in CONTEXT_GROUPS.values() for word in group["new"]}
    assert all(item.anchor in new_words for item in test)
    assert all(item.anchor not in new_words for item in training)
    assert all(any(word in item.anchor for word in new_words) for item in training)
