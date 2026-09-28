# Research log

This log records hypothesis changes as results arrived. It intentionally keeps
negative outcomes; E1 is an experiment, not a sequence of demos.

## E1 smoke — pipeline

Implemented grapheme events, recurrent ALIF layers, surrogate gradients,
multiscale Van Rossum distance, triplet loss, post-stimulus activity, and
sparsity regularization. The 18k-parameter CPU model could fit the IID task
without collapse.

## E1.1 — output timing controls

Five seeds reproduced semantic triplet performance. Local/global spike-time
shuffles and rate-only evaluation showed that precise output timing was not
necessary. H3b was rejected for this regime.

## E1.2A — metric sensitivity and depth

The slow Van Rossum scale (`τ=14`) matched or exceeded the multiscale metric;
H4a was not supported. Layer-wise margins revealed increasingly semantic
separation with depth.

## E1.2C — input sequence destruction

Reverse, permutation, bag-of-graphemes, and controlled jitter preserved input
event counts. Destroying order strongly reduced performance, while mild tempo
changes caused smaller graded losses. H3a was reframed as ordered/sequential
dependency—not intrinsic precise timing.

## E1.2B — neuronal time scales

Four membrane-decay configurations showed no conclusive global advantage, but
learnable heterogeneous dynamics improved hard orthographic separation versus
one fixed β. Diagnostics found little learned L3 β specialization, suggesting
the initialization supplies most temporal diversity.

## Factorial pilot — form to semantics

A curated orthography×semantics matrix produced a population-code abstraction
index `−0.854 → −0.413 → +0.188`. The result is the central internal-analysis
figure, with explicit lexical-sampling and exposure caveats.

## Paired ANN baseline

An exactly parameter-matched adaptive leaky recurrent ANN also showed the
factorial inversion and achieved better global/generalization results. The
phenomenon is therefore hierarchical, not uniquely spiking. The SNN retained
greater sparsity, order dependence, and hard-orthographic accuracy.

## Generalization

Relation-disjoint testing exceeded chance for both models. Context-supervised
exposure to new lexemes transferred to isolated-word tests, with matched-update
controls. This closed the immediate identifiability gap but did not constitute
unsupervised distributional learning.

## Current hypothesis map

```text
G0 Pipeline                                    supported
G1 Micro-learning                              supported
G2 IID reproducibility                         supported
H1 Supervised semantic organization            supported in small regime
H2 Form→semantic shift across depth             supported diagnostically
H3a Ordered/sequential dependency               supported
H3b Precise output spike timing                 not supported
H3c Tempo invariance                            open
H4a Multiscale Van Rossum advantage             not supported
H4b Neuronal multiscale advantage               partial evidence
H5 Relation/context-exposed generalization      supported narrowly
SNN-specific hierarchy                          not supported (ANN also shows it)
```

