# E1-v2 frozen release

Release date: 2026-09-30  
Release version: `2.0.0-e1`  
Git tag: `v2.0-e1`

## Scope

E1-v2 preserves the original E1 training runs and extends their analysis with
word-aligned post-stimulus settling, trajectory-specific readouts, and causal
interventions on frozen SNN and ANN checkpoints. The internal experiment name
`E1.5` is retained in result manifests to preserve provenance; `E1-v2` denotes
the paper and reproducibility release that incorporates that extension.

## Frozen manuscript

`paper/orthographic-to-semantic-abstraction-e1-v2.pdf`

SHA-256:
`e80c34394dfa76d468bed7a1db3613ed5c1d35d75b7af214cb206fa689639526`

The release manuscript has seven pages. It reports the SNN layer-3 integrated
trajectory estimate at `t+72` as `A3 = +0.163`, with a seed-bootstrap 95%
interval `[+0.037, +0.292]`, and operationally defines all four post-stimulus
interventions.

## Reproduction surface

- `make test` validates the implementation and data invariants.
- `make mechanistic` reproduces the frozen-checkpoint settling analysis.
- `runs/mechanistic/manifest.json` records configuration, runtime, dataset,
  checkpoint, and output hashes for the paper-facing mechanistic run.
- `docs/REPRODUCIBILITY.md` documents the complete commands and limitations.

No E2-v2 data, models, or results are part of this release.
