from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import __version__


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(cwd: str | Path = "."):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=cwd, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return {"commit": commit, "branch": branch or None, "dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def artifact_record(path: str | Path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def write_manifest(
    output_dir: str | Path,
    *,
    experiment: str,
    config_path: str | Path,
    seeds: list[int],
    datasets: list[str | Path],
    outputs: list[str | Path],
    checkpoints: list[str | Path] | None = None,
    extra: dict | None = None,
):
    output_dir = Path(output_dir)
    config_path = Path(config_path)
    manifest = {
        "schema_version": 1,
        "experiment": experiment,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "code": {"package_version": __version__, **git_revision()},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "config": artifact_record(config_path),
        "datasets": [artifact_record(path) for path in datasets],
        "checkpoints": [artifact_record(path) for path in (checkpoints or [])],
        "outputs": [artifact_record(path) for path in outputs if Path(path).exists()],
        "extra": extra or {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
