# Reproducibility

## Environment

- Python `>=3.11,<3.14`
- PyTorch CPU build selected through `pyproject.toml`
- No GPU required
- Deterministic experiment seeds: `7, 19, 31, 43, 59`

```bash
uv sync --extra dev
uv run pytest
```

## Quick verification

```bash
./scripts/reproduce_quick.sh
```

This runs unit tests and a three-epoch smoke experiment. It verifies mechanics,
not the reported confidence intervals.

## Complete sequence

```bash
./scripts/reproduce_all.sh
```

The sequence is intentionally explicit:

1. E1.1 checkpoints and temporal controls.
2. E1.2A frozen-checkpoint metric analysis.
3. E1.2C input-order corruptions.
4. E1.2B neuronal time-scale ablations.
5. β diagnostics and factorial evaluation.
6. Paired ANN training/evaluation.
7. Generalization protocols.
8. Consolidated report and central SVG figure.

Expect approximately 15–25 CPU minutes, depending on the processor. Per-seed
JSON files and checkpoints are written to `runs/`. Compact report artifacts are
written to `reports/`.

## Statistical unit

Headline confidence intervals bootstrap five random-seed results. Lexical pairs
are fixed. These intervals must not be interpreted as uncertainty over the
Spanish lexicon or over alternative dataset construction choices.

## Dataset integrity

- `data/factorial_pairs.jsonl` records relation type, part of speech, and
  supervision exposure for every curated pair.
- Relation-disjoint tests remove both orientations of every held positive edge.
- Context-transfer targets never appear as isolated training inputs.
- The no-context control receives the same number of optimizer examples.

