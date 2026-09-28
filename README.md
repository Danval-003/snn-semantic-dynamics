# E1 — Hierarchical orthographic-to-semantic organization in a recurrent SNN

E1 is a small, CPU-reproducible study of whether a recurrent spiking neural
network can transform grapheme events into semantically organized neural
activity without BPE, word embeddings, attention, Transformers, or pretrained
semantic representations.

The current model has **18,128 parameters**. Its forward pass is genuinely
spiking; semantic supervision acts directly on population spike trains through
a multiscale Van Rossum triplet loss.

> Current result: ordered sequential processing is required to construct the
> representation, while the final semantic organization is expressed mainly
> as a sparse population/rate code rather than precise output spike timing.

This is supervised semantic organization—not unsupervised semantic emergence.

![Factorial orthographic-to-semantic transformation](reports/figures/factorial_abstraction.svg)

## Main findings

- The SNN reaches `0.913` IID population-code triplet accuracy across five seeds.
- Shuffling final spike timing does little; rate-only retains the result.
- Permuting input graphemes reduces SNN accuracy from `0.913` to `0.654`.
- A curated 2×2 diagnostic changes from orthographic dominance in L1/L2 to
  semantic dominance in the L3 rate code: `A = −0.854 → −0.413 → +0.188`.
- Heterogeneous learnable membrane dynamics outperform one fixed time constant
  on hard orthographic distractors: `0.640 vs 0.490` temporal STA.
- Relation-disjoint SNN generalization reaches `0.778`.
- Context-supervised exposure transfers new lexemes to isolated-word tests:
  `0.487 → 0.758` for the SNN.

## Paired continuous ANN baseline

The ANN uses the same input, depth, widths, recurrence, adaptation, temporal
parameters, training data, seeds, and **18,128 parameters**. Binary spikes are
replaced by continuous adaptive leaky-tanh states.

| Metric | SNN | ANN | Paired interpretation |
|---|---:|---:|---|
| IID population STA | 0.913 | **0.928** | No conclusive difference |
| Hard orthographic STA | **0.720** | 0.540 | SNN +0.180 [0.040, 0.360] |
| Factorial abstraction `A₃` | **0.188** | 0.141 | No conclusive difference |
| Accuracy drop after input permutation | **0.259** | 0.133 | SNN is more order-dependent |
| Relation-disjoint STA | 0.778 | **0.906** | ANN generalizes better |
| Contextual lexical transfer | 0.758 | **0.833** | ANN generalizes better |

The layer-wise form→semantics transformation is therefore **not unique to
spikes**. The SNN's current distinctive properties are sparsity, stronger order
dependence, and better separation of orthographically deceptive examples. A
spike rate of `0.090` versus `0.986` ANN states above `|10⁻³|` is operationally
informative but is not an energy comparison.

## Architecture

```text
grapheme events (one-hot channels, fixed timing)
        ↓
recurrent adaptive spiking layer — fast
        ↓
recurrent adaptive spiking layer — intermediate
        ↓
recurrent adaptive spiking layer — slow/conceptual
        ↓
binary population trajectory [time, neurons]
```

Each layer has recurrent connections, adaptive thresholds, and heterogeneous
learnable membrane decay. A silent post-stimulus window measures persistent
state. The paired ANN mirrors this hierarchy with continuous units.

## Reproduce

Requirements: Linux/macOS, Python 3.11–3.13, and `uv`. PyTorch is pinned to its
CPU-only index; no GPU is required.

```bash
uv sync --extra dev
uv run pytest
uv run e1-smoke --epochs 5
```

Convenience commands:

```bash
make test       # unit tests
make quick      # tests + short smoke run
make full       # all experiments, summaries, and figures
make figures    # regenerate tracked SVG figures from run artifacts
```

The full suite runs five seeds and takes roughly 15–25 minutes on a typical
CPU. Generated checkpoints and detailed per-seed artifacts live under `runs/`
and are ignored by Git. Compact results and figures under `reports/` are
versioned.

## Repository map

```text
configs/                  experiment configurations
data/                     audited, versioned evaluation pairs
docs/EXPERIMENT.md        full protocol and detailed results
docs/RESEARCH_LOG.md      chronological research decisions
docs/RESULTS.md           frozen result tables and claim boundaries
docs/PAPER_OUTLINE.md     working six-page manuscript outline
docs/REPRODUCIBILITY.md   protocols, runtimes, and exact commands
reports/                  compact results and tracked figures
scripts/                  complete and quick reproduction entry points
src/e1_spikes/            models, metrics, controls, and runners
tests/                    mechanical and split-integrity tests
```

## Claim boundaries

E1 currently supports a layer-wise shift from orthographic to semantic
organization under supervised training. It does **not** establish unsupervised
language acquisition, broad lexical generalization, energy efficiency,
reasoning, text generation, or superiority of SNNs over ANNs. The contextual
transfer experiment is semantically supervised at the sentence level; it is
not unsupervised corpus exposure.

See [the frozen results](docs/RESULTS.md), [the research log](docs/RESEARCH_LOG.md),
and [the full protocol](docs/EXPERIMENT.md).

## License

Released under the [MIT License](LICENSE).
