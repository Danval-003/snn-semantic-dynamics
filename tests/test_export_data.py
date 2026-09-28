import json

from e1_spikes.export_data import rendered_files


def test_exports_are_deterministic_and_manifested():
    first = rendered_files()
    second = rendered_files()
    assert first == second
    manifest = json.loads(first["manifest.json"])
    assert manifest["seed"] == 20260928
    assert manifest["files"]["fixed_validation.jsonl"]["examples"] == 260
    assert manifest["files"]["relation_disjoint_test.jsonl"]["examples"] == 36
    assert manifest["files"]["context_transfer_test.jsonl"]["examples"] == 48


def test_relation_disjoint_edges_do_not_leak_into_training():
    files = rendered_files()
    training = [json.loads(line) for line in files["relation_train.jsonl"].splitlines()]
    test = [json.loads(line) for line in files["relation_disjoint_test.jsonl"].splitlines()]
    training_edges = {
        tuple(sorted((row["anchor"], row["positive"]))) for row in training
    }
    test_edges = {tuple(sorted((row["anchor"], row["positive"]))) for row in test}
    assert training_edges.isdisjoint(test_edges)
