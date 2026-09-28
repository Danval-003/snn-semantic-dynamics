# Working paper outline

Provisional title:

> Hierarchical Orthographic-to-Semantic Organization in Small Recurrent
> Spiking Neural Networks

Use “semantic organization” or “semantic separation,” not unsupervised
“semantic emergence.”

## 1. Introduction

- Language models normally convert discrete symbols into continuous embeddings.
- Ask whether a recurrent SNN can organize semantic supervision directly in
  population activity without a token embedding as final representation.
- Contribution: controlled internal-analysis study, not a complete language model.

## 2. Related work

- Spiking language and sequence models.
- Temporal versus population coding.
- Hierarchical representation analysis.
- Character-level semantic encoders and recurrent baselines.

This section still requires a formal literature search and citations.

## 3. Method

- Grapheme-event encoder and minimal normalization.
- Three-layer recurrent adaptive SNN.
- Surrogate-gradient training.
- Van Rossum triplet objective and activity regularization.
- Exactly parameter-matched continuous ANN.

## 4. Experimental design

- Fixed IID validation and hard orthographic negatives.
- Five seeds and seed-level bootstrap intervals.
- Output-time shuffles and rate-only readout.
- Input order/jitter corruptions.
- Neuronal time-scale ablation.
- Curated 2×2 orthography×semantics diagnostic.
- Relation-disjoint and context-exposed transfer protocols.

## 5. Results

1. Semantic supervision is learned without collapse.
2. Final coding is predominantly population/rate based.
3. Ordered input processing is necessary.
4. Orthographic dominance decreases and semantic sensitivity increases by depth.
5. Temporal heterogeneity helps against hard distractors versus one fixed β.
6. ANN reproduces the hierarchy and generalizes better globally.
7. SNN is sparse and stronger on hard orthographic distractors.
8. Context-supervised lexical transfer succeeds in both models.

Central figure: `reports/figures/factorial_abstraction.svg`.

## 6. Discussion

Core distinction:

> Time acts primarily as a computational mechanism for constructing semantic
> organization, not necessarily as the final representational format.

Do not claim SNN superiority. Discuss the observed trade-off: ANN
generalization versus SNN sparsity/order dependence/hard-negative behavior.

## 7. Limitations

- Small curated Spanish vocabulary and supervised relations.
- Factorial cells differ in relation type and exposure.
- Context exposure is semantically supervised, not unsupervised.
- No lexical-frequency control or external human-similarity benchmark yet.
- Operational activity is not hardware energy.
- No audio, generation, or reasoning.
- Confidence intervals cover seeds, not lexical-dataset sampling.

## Remaining work before a submission

- Replicate the factorial effect on a larger, externally sourced lexical set.
- Add a literature review and clearly positioned related work.
- Decide whether H3c tempo augmentation belongs in this paper or follow-up work.
- Produce publication-quality figures with lexical-item bootstrap/sensitivity.
- Package exact environment and archive artifacts with a DOI.

