.PHONY: setup test data quick full figures mechanistic

setup:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv sync --extra dev

test:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run pytest

data:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run e1-export-data

quick:
	./scripts/reproduce_quick.sh

full:
	./scripts/reproduce_all.sh

figures:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run python scripts/make_figures.py

mechanistic:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run e1-mechanistic --config configs/mechanistic.toml
