import json

from e1_spikes.provenance import file_sha256, write_manifest


def test_manifest_hashes_config_datasets_and_outputs(tmp_path):
    config = tmp_path / "config.toml"
    dataset = tmp_path / "data.jsonl"
    output = tmp_path / "seed_7.json"
    checkpoint = tmp_path / "seed_7.pt"
    config.write_text("seed = 7\n", encoding="utf-8")
    dataset.write_text('{"x": 1}\n', encoding="utf-8")
    output.write_text('{"ok": true}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    manifest = write_manifest(
        tmp_path, experiment="test", config_path=config, seeds=[7],
        datasets=[dataset], outputs=[output], checkpoints=[checkpoint],
    )
    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert stored == manifest
    assert stored["datasets"][0]["sha256"] == file_sha256(dataset)
    assert stored["outputs"][0]["sha256"] == file_sha256(output)
    assert stored["checkpoints"][0]["sha256"] == file_sha256(checkpoint)
