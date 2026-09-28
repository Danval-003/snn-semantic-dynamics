#!/usr/bin/env bash
set -eu

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/e1-uv-cache}"
uv sync --extra dev
uv run e1-export-data
uv run pytest
uv run e1-e11
uv run e1-e12a
uv run e1-e12c
uv run e1-e12b
uv run e1-beta-diagnostics
uv run e1-factorial
uv run e1-ann-baseline
uv run e1-generalization
uv run e1-consolidate
uv run python scripts/make_figures.py
