# Frozen results — E1

All headline values are means across seeds `7, 19, 31, 43, 59`. Confidence
intervals are seed-level percentile bootstrap intervals. They describe
run-to-run variation, not uncertainty over lexical-item sampling.

## Central factorial diagnostic

The curated set contains 12 pairs per cell. `A = S − O`, where `S` is relative
semantic sensitivity and `O` relative orthographic sensitivity.

| Readout | L1 | L2 | L3 |
|---|---:|---:|---:|
| SNN Van Rossum `A` | −0.911 | −0.571 | +0.035 |
| SNN population/rate `A` | −0.854 | −0.413 | **+0.188** |
| ANN integrated-state `A` | −0.760 | −0.682 | **+0.141** |

SNN population L3: `A=0.188 [0.075, 0.304]`. ANN L3:
`A=0.141 [0.064, 0.219]`. Their paired difference is not conclusive.

The diagnostic is not a causal factorial experiment: cells differ in exposure
and relation type, frequency is uncontrolled, and lexical sampling is small.

## SNN temporal controls

| Control | STA |
|---|---:|
| Original Van Rossum | 0.910 |
| Local shuffle ±1 | 0.910 |
| Local shuffle ±3 | 0.913 |
| Global output-time shuffle | 0.886 |
| Rate-only | 0.913 |

Precise final spike timing is not supported as the semantic code. Conversely,
input order is important: rate STA falls from `0.913` to `0.654` after grapheme
permutation and to `0.640` for a bag-of-graphemes corruption.

## Neuronal time-scale ablation

| Membrane dynamics | Global STA | Hard STA | Hard L3 margin |
|---|---:|---:|---:|
| One fixed β | 0.892 | 0.490 | 0.029 |
| Fixed β by layer | 0.900 | 0.520 | 0.086 |
| Fixed heterogeneous β | 0.885 | 0.520 | 0.126 |
| Learnable heterogeneous β | **0.910** | **0.640** | **0.156** |

Learnable heterogeneous β beats one fixed β on hard STA by `+0.150
[0.020, 0.280]`, but does not conclusively beat fixed layer-wise or fixed
heterogeneous dynamics. Per-neuron diagnostics show little learned β
specialization in L3; the heterogeneous initialization likely provides most of
the useful structure.

## SNN vs paired ANN

| Metric | SNN | ANN | SNN−ANN, paired 95% CI |
|---|---:|---:|---:|
| IID population STA | 0.913 | 0.928 | −0.015 [−0.056, 0.012] |
| Hard orthographic STA | **0.720** | 0.540 | +0.180 [0.040, 0.360] |
| Factorial `A₃` | 0.188 | 0.141 | +0.046 [−0.103, 0.142] |
| Input-permutation drop | **0.259** | 0.133 | +0.127 [0.059, 0.166] |
| Relation-disjoint STA | 0.778 | **0.906** | −0.128 [−0.200, −0.056] |
| Contextual lexical transfer | 0.758 | **0.833** | −0.075 [−0.142, −0.008] |

The ANN baseline rules out a spikes-specific claim for hierarchical semantic
organization. It also generalizes better. The SNN is sparse and performs better
on orthographically deceptive examples, motivating—not proving—further study.

## Generalization protocols

### Relation-disjoint

Both lexemes occur in training, but each tested semantic edge and its reverse
are excluded. Population STA: SNN `0.778 [0.706, 0.844]`; ANN `0.906
[0.850, 0.961]`.

### Context-exposed lexical transfer

Eight new lexemes occur only inside supervised semantic context triplets and
are then evaluated as isolated strings. A matched control receives the same
number of updates without contexts.

| Model | No context | Context exposed | Paired gain |
|---|---:|---:|---:|
| SNN | 0.487 | 0.758 | +0.271 [0.208, 0.333] |
| ANN | 0.546 | 0.833 | +0.288 [0.254, 0.313] |

This demonstrates supervised contextual transfer to isolated inputs. It does
not demonstrate unsupervised acquisition from raw text.

Machine-readable paired comparisons are in
[`reports/benchmark_summary.json`](../reports/benchmark_summary.json).

