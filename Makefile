.PHONY: setup test quick full figures

setup:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv sync --extra dev

test:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run pytest

quick:
	./scripts/reproduce_quick.sh

full:
	./scripts/reproduce_all.sh

figures:
	UV_CACHE_DIR=/tmp/e1-uv-cache uv run python scripts/make_figures.py

