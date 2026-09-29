# E1.5 — post-stimulus recurrent settling

## Scope

This is an analysis-only follow-up on the frozen E1 checkpoints. No model is
retrained and no hyperparameter is selected from these results. Every temporal
offset is measured relative to each example's final real grapheme event.

The runner evaluates SNN and matched ANN checkpoints under four interventions:

- `natural`: unmodified recurrent evolution;
- `no_recurrent`: recurrent communication is removed only after the stimulus;
- `intrinsic_only`: recurrent and feed-forward drive are removed after the
  stimulus, while leak and adaptation remain;
- `reset_state`: the recurrent state is cleared immediately after the stimulus.

Five representations are stored independently: `stimulus_only`,
`instantaneous`, `local_window`, `post_only`, and `full_trajectory`.

## Legacy parity

For every model and seed, the runner checks:

1. legacy and explicit encoders produce identical event tensors;
2. the natural forward produces bit-identical layer states;
3. the integrated legacy representation is identical.

All 10 checkpoint checks pass. The v1.0 execution path therefore remains a
valid compatibility mode rather than an approximate reimplementation.

## Natural settling

Layer-3 `full_trajectory` results:

| Model | Offset | STA | Hard STA | Original factorial \(A_3\) | Lexical holdout \(A_3\) |
|---|---:|---:|---:|---:|---:|
| SNN | t+0 | .832 | .560 | −.221 | −.357 |
| SNN | t+12 | .908 | .690 | +.185 | +.154 |
| SNN | t+72 | .914 | .720 | +.163 | +.154 |
| ANN | t+0 | .647 | .190 | −1.151 | −.844 |
| ANN | t+12 | .855 | .350 | −.263 | −.035 |
| ANN | t+72 | .905 | .470 | +.002 | +.304 |

For the SNN, the paired t+0→t+12 changes are:

- STA `+.076` `[+.035, +.108]`;
- hard STA `+.130` `[+.040, +.230]`;
- original factorial \(A_3\): `+.407` `[+.368, +.455]`;
- lexical holdout \(A_3\): `+.511` `[+.399, +.644]`.

Thus the sign change is not restricted to the original 48 curated pairs. The
holdout is nevertheless a fixed second lexical sample, not an estimate of
uncertainty over Spanish. Its semantic-positive, orthographically-different
cell includes concept relations rather than only strict synonyms, so effect
magnitudes are not directly interchangeable between datasets.

## Causal interventions at t+12

Natural minus intervention, paired over seeds:

| Model | Intervention | ΔSTA | ΔHard | Δ original \(A_3\) | Δ holdout \(A_3\) |
|---|---|---:|---:|---:|---:|
| SNN | no recurrent | +.018 | +.020 | +.139 | +.164 |
| SNN | intrinsic only | +.033 | +.060 | +.170 | +.250 |
| SNN | reset state | +.076 | +.130 | +.407 | +.511 |
| ANN | no recurrent | +.041 | +.140 | +.239 | +.162 |
| ANN | intrinsic only | +.143 | +.110 | +.581 | +.554 |
| ANN | reset state | +.208 | +.160 | +.888 | +.808 |

For both datasets, the SNN natural−no-recurrent \(A_3\) interval is positive:
original `+.139` `[+.090, +.193]`, holdout `+.164` `[+.086, +.242]`.
Post-stimulus recurrent communication therefore contributes causally to the
development of the integrated semantic geometry. Intrinsic dynamics also
contribute; recurrence is not the sole mechanism.

## Readout distinction

At SNN t+12, original-factorial \(A_3\) is negative in `stimulus_only`
(`−.221`), `instantaneous` (`−.499`), `local_window` (`−.135`), and `post_only`
(`−.271`), but positive in `full_trajectory` (`+.185`). The supported claim is:

> Semantic separation in the integrated population trajectory continues to
> develop during stimulus-free recurrent evolution.

This does not establish an attractor, deliberation, thought, or a semantic
instantaneous late state.

The ANN shows a different late profile: the original factorial approaches zero
at t+72 while the lexical holdout becomes positive. Because the two lexical
sets differ in relation composition, this is a target for replication rather
than evidence of a general ANN semantic crossing.

## Artifacts

- configuration: `configs/mechanistic.toml`;
- aggregate and per-seed metrics: `runs/mechanistic/`;
- provenance: `runs/mechanistic/manifest.json`;
- frozen lexical set: `data/generated/lexical_holdout_pairs.jsonl`.
