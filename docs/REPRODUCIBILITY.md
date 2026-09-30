# Reproducibility

## Environment

- Python `>=3.11,<3.14`
- PyTorch CPU build selected through `pyproject.toml`
- No GPU required
- Deterministic experiment seeds: `7, 19, 31, 43, 59`

```bash
uv sync --extra dev
uv run e1-export-data --check
uv run pytest
uv run e1-mechanistic --config configs/mechanistic.toml
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

1. Regenerate the tracked dataset manifests.
2. Run the test suite.
3. E1.1 checkpoints and temporal controls.
4. E1.2A frozen-checkpoint metric analysis.
5. E1.2C input-order corruptions.
6. E1.2B neuronal time-scale ablations.
7. β diagnostics and factorial evaluation.
8. Paired ANN training/evaluation.
9. Generalization protocols.
10. Consolidated report, raw metrics, and central SVG figure.

Expect approximately 15–25 CPU minutes, depending on the processor. Per-seed
JSON files and checkpoints are written to `runs/`. Compact report artifacts are
written to `reports/`. The tracked `reports/raw_metrics.jsonl` contains one row
per model and seed for every headline comparison, so confidence intervals can
be recomputed without rerunning training.

## Statistical unit

Headline confidence intervals bootstrap five random-seed results. Lexical pairs
are fixed. These intervals must not be interpreted as uncertainty over the
Spanish lexicon or over alternative dataset construction choices.

## Dataset integrity

- `data/factorial_pairs.jsonl` records relation type, part of speech, and
  supervision exposure for every curated pair.
- `data/generated/` exposes the exact fixed validation, relation-training,
  relation-disjoint, context-exposure, context-transfer, and matched-control
  triplets as JSONL.
- `data/generated/manifest.json` records row counts and SHA-256 digests.
- `data/generated/lexical_holdout_pairs.jsonl` is a second balanced 2×2 set,
  disjoint from the original factorial and excluded from all training.
- `uv run e1-export-data --check` fails if the tracked exports diverge from the
  constructors used by the experiment; the CI workflow runs this check.
- Relation-disjoint tests remove both orientations of every held positive edge.
- Context-transfer targets never appear as isolated training inputs.
- The no-context control receives the same number of optimizer examples.

## Versioned artifact

The manuscript is stored at
[`paper/orthographic-to-semantic-abstraction-e1-v2.pdf`](../paper/orthographic-to-semantic-abstraction-e1-v2.pdf).
The original E1-v1 manuscript remains frozen at
[`paper/orthographic-to-semantic-abstraction.pdf`](../paper/orthographic-to-semantic-abstraction.pdf)
and in tag `v1.0-e1`.
The E1 artifact is cited through [`CITATION.cff`](../CITATION.cff) and released
under the MIT License. Release tags identify frozen paper-facing snapshots;
generated checkpoints remain excluded because they are reproducible and are
not required to audit the reported metrics.

## Mechanistic extension and run manifests

E1.5 separates training from analysis: `e1-mechanistic` loads the frozen SNN
and ANN checkpoints and runs all readouts, offsets, and interventions without
updating parameters. Use `--summarize-only` to recompute aggregate intervals
from per-seed JSON files without loading a checkpoint.

Every E1.5 run writes `runs/mechanistic/manifest.json` with the exact config and
dataset and checkpoint SHA-256 hashes, seeds, package/Python/PyTorch versions,
Git commit and dirty state, and hashes of every result file. Historical v1.0
runners are left unchanged to protect paper parity; the manifest contract
applies to the E1-v2 mechanistic runners.
