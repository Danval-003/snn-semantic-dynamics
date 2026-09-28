#!/usr/bin/env bash
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/e1-uv-cache}"
uv sync --extra dev
uv run e1-export-data --check
uv run pytest
uv run e1-smoke --epochs 3
