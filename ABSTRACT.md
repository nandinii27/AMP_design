# Abstract

**Composition-limited peptide design: a generative pipeline with 
evaluation and a factorially designed validation batch**

I submit a generative model for antimicrobial peptide design together with four
negative results that constrain what sequence-derived descriptors can contribute
to this task.

**Method.** A character-level transformer language model (809k parameters,
20 epochs) trained on the 29,023 cysteine-free members of the challenge
reference set proposes candidates; four gradient-boosted regressors predict
log2 MIC against Gram-positive and Gram-negative panels, log2 HC50 on human
erythrocytes, and log2 MIC against resistant clinical isolates. Candidates are
ranked by augmented Chebyshev scalarisation over four objectives — Gram-positive
potency, Gram-negative potency, safety window, and a synthesis-survival term —
and refined by register-aware simulated annealing whose operators are aware of
alpha-helical geometry. A robust Mahalanobis gate rejects candidates outside the
descriptor envelope of validated actives, which prevents the search from
exploiting regions where the oracle extrapolates.

**Scoring-rule analysis.** The competition scores the arithmetic mean over
twenty-five peptides drawn at random from a team's top fifty. Because the score
is a mean rather than a maximum, specialist portfolios are strictly dominated:
ten Gram-negative specialists at 0.9 mixed with fifteen peptides at 0.1 score
0.42 in the Gram-negative category, while twenty-five balanced peptides at 0.5
score 0.50. Targeting all five categories is therefore a per-peptide
scalarisation problem rather than a portfolio allocation one, which motivates the
Chebyshev formulation: maximising the weakest normalised objective directs
optimisation pressure wherever a candidate is currently worst.

**Negative results.** Under homology-controlled evaluation (greedy clustering at
40% identity, five repeated splits), no family of sequence-derived structural
descriptor improved prediction over amino acid composition. On HC50 (n = 5,550),
composition alone reached Spearman 0.498 +/- 0.057; adding hydrophobic moment and
spectral purity gave 0.505, circular helical-wheel statistics 0.504, and H0
persistence of the three-dimensional hydrophobic point cloud 0.508 +/- 0.068. The
result replicated on Gram-negative MIC (n = 11,275, 16% censored versus 62%):
0.505 to 0.521 across the same tiers. A conditional-disorder block modelling the
coil-to-helix transition — Lifson-Roig helicity in water and at a membrane-mimetic
interface, their difference, and an explicit non-monotone switchability term —
changed held-out performance by -0.010 while consuming 13.4% of total tree gain,
the signature of features correlated with information the model already has.

**Evaluation-protocol ablation.** A two-by-two over split type and censoring
treatment quantified inflation from evaluation choices at +0.072 Spearman on
Gram-negative MIC, decomposing into +0.021 from homology leakage and +0.051 from
recording assay ceilings as point measurements. The second effect is larger than
the first and has received less attention. I noted that reported haemolysis
values pile up at round levels (100 appears 1,244 times as a bare value against
254 times as ">100"), and treat bare values at standard ceilings as
right-censored. Variance is also informative: random splits gave +/- 0.004 across
repeats and clustered splits +/- 0.049, so tight error bars under random splitting
measure the stability of leakage rather than generalisation.

**Designed validation batch.** Because the literature cannot distinguish "the
descriptors are uninformative" from "pooled multi-laboratory data is too noisy to
resolve them", I select the draw pool as a 2x2x2 factorial over hydrophobic
moment, hydrophobic-face patch count, and net charge, within a tolerance band in
which predicted scores are not meaningfully distinguishable. The design costs
3.87% of mean predicted score relative to greedy selection. A single-laboratory
measurement under one protocol is the instrument that can separate the two
explanations, and the batch is constructed so that it will.

**Availability.** MIT-licensed, `uv`-managed, seeded for byte-identical
reproduction, verified with the organisers' harness.
