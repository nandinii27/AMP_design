# Selection and ranking procedure

How 50,000 library sequences become a ranked list of 100 candidates.

## Why the objective has the shape it does

The competition scores the arithmetic mean over twenty-five peptides drawn at
random from a team's top fifty. Two consequences follow, and both drive the
design.

**Specialisation is strictly dominated.** Because the score is a mean rather than
a maximum, a portfolio of ten Gram-negative specialists at success rate 0.9 mixed
with fifteen peptides at 0.1 scores (10 x 0.9 + 15 x 0.1) / 25 = 0.42 in the
Gram-negative category. Twenty-five balanced peptides at 0.5 score 0.50. The
generalist portfolio wins in the specialist's own category. Targeting all five
categories is therefore a per-peptide scalarisation problem, not a portfolio
allocation problem.

**Every slot carries equal weight, and correlated failure is the dominant risk.**
Expectation is linear over the draw, so position fifty matters exactly as much as
position one. If the fifty are variations on one scaffold, a single shared
liability — an aggregation-prone motif, a synthesis failure mode, one wrong model
prediction — removes many slots at once. Diversity here is variance reduction
rather than presentation.

## The objective

Augmented Chebyshev scalarisation over four axes:

```
score(x) = min_k [ w_k * f_k(x) / ideal_k ]  +  rho * mean_k [ f_k(x) / ideal_k ]
```

with a multiplicative physicochemical factor applied afterwards.

| Axis | Definition |
|---|---|
| Gram-positive potency | P(MIC <= 16 uM), smoothed by the predictive spread |
| Gram-negative potency | as above |
| Safety window | predicted log2 HC50 minus the better predicted log2 MIC |
| Synthesis survival | product of length, hydrophobic-run, oxidation and motif terms |

**Chebyshev rather than a weighted sum**, because a weighted sum permits
compensation: the search abandons the hardest objective to buy gains in the
easiest. The `min` directs optimisation pressure wherever a candidate is
currently weakest. In practice that is almost always the safety window, since
potency is comparatively easy to reach — so the selectivity emphasis falls out of
the formulation rather than being hard-coded. This is visible in the output: the
final top-100 has a safety-window standard deviation of 0.24, with every
candidate pushed against the same ceiling.

**Targets are set at quantiles of the labelled actives, not at the competition
floor.** Fifty-three percent of DBAASP actives already clear 16 uM, so optimising
to that bar is unambitious. The safety-window ideal is the 75th percentile of the
paired MIC/HC50 distribution (5.06 in log2 units).

**Synthesis survival multiplies rather than adjusts rank.** A candidate that
fails synthesis, proves insoluble or misses purity control is not retested and
its slot in the experimental batch is simply lost, so this is an expected-value
term.

**Soft constraints are multiplicative, hard constraints are rejections.** Under
an additive penalty, a candidate two charge units outside the design window with
predicted quality 12 beats a compliant candidate at quality 9, so the optimiser
trades the constraint away and lands where the model extrapolates. A
multiplicative Gaussian factor decays faster than any quality term can grow.
Alphabet, length, uniqueness, cysteine exclusion and the identity ceiling are
enforced by rejection and are never traded against quality.

**Design windows were calibrated, not assumed.** Literature priors of charge
[4, 8] were checked against the reference set and found to sit at median 2.85
with only 23.5% inside the window. Refitting to that full distribution would have
been a different error, since roughly a quarter of it is cysteine-rich
disulfide-stabilised peptides this design excludes. The windows were therefore
refit on the correct subpopulation — potent (MIC <= 16 uM), cysteine-free —
giving charge median 4.99 with a 10th-90th percentile band of [2.0, 8.0].

## The screening cascade

Cheap filters at the wide end, expensive ones only at the narrow end. The
organisers' harness runs generation twice on their hardware, so the runtime
budget is real.

| Stage | Survivors | Filter |
|---|---|---|
| 0 | 69,343 | legality, uniqueness, not identical to a known peptide |
| 1 | 20,000 | in-distribution gate (92.5% pass rate) |
| 2 | 2,000 | Chebyshev objective |
| 3 | 54,526 | register-aware annealing from 150 seeds |
| 4 | 100 | novelty filter, then diverse subset selection |

**The in-distribution gate is the anti-reward-hacking mechanism.** A robust
Mahalanobis screen (minimum covariance determinant, 97.5th percentile) fitted on
5,989 potent cysteine-free peptides. Without it, any search against a learned
oracle drifts toward the corner of descriptor space where the oracle extrapolates
most confidently and has seen nothing. It is treated as a hard constraint: a
candidate outside the envelope is not worse, it is unscoreable.

**The library is an output of the pipeline, not its input.** The local search
produces sequences that were never sampled, and the harness requires every
top-list member to appear in the library. Refined candidates are therefore placed
in the library first and the remainder filled from the sampled pool, which makes
that requirement true by construction.

**Search operators know the helix geometry.** An alpha-helix advances 100 degrees
per residue, so positions three or four apart occupy the same face. Substitutions
are proposed differently depending on which face a position lies on, and a
composition-preserving swap operator changes arrangement while holding net
charge, mean hydrophobicity and mass fixed. Insertions and deletions are kept
rare because a single indel rotates the register of every downstream residue.

**The 80% identity ceiling rules out template engineering.** A single
substitution on a 20-mer leaves 95% identity; three leave 85%. Candidates require
at least four edits from every one of the 39,448 reference peptides, so the
search starts from de novo language-model samples and never from natural
templates.

## Ranking within the top 100

Ranked by risk-adjusted Chebyshev score: the objective less a modest penalty on
ensemble spread, since model error is correlated across similar sequences and the
realised batch mean therefore has real variance. Most variance reduction comes
from portfolio diversity rather than from a large risk coefficient.

Positions 1-50 are the draw pool and carry the experimental design. Positions
51-100 are insurance against identity-filter rejection and are filled by score
and divergence.

## The factorial draw pool

Four ablations in this repository found that no family of sequence-derived
structural descriptor improves prediction over amino acid composition under
homology-controlled evaluation (details in the abstract). Those results cannot
distinguish two explanations: either arrangement genuinely does not matter, or
pooled multi-laboratory data is too noisy to resolve it.

A single-laboratory measurement under one protocol is the instrument that
separates them. The draw pool is therefore selected as a 2x2x2 factorial over
hydrophobic moment, hydrophobic-face patch count (H0 persistence of the
three-dimensional hydrophobic point cloud), and net charge, with six peptides per
cell, each cell split at the median of the eligible pool.

Selection operates within a tolerance band in which predicted scores are not
meaningfully distinguishable: with held-out Spearman near 0.5 and
within-one-dilution accuracy near 0.4, differences inside the band are below the
resolution of the models. The band contained 1,003 candidates. The design costs
**3.87%** of mean predicted score relative to greedy selection, measured rather
than assumed, and the implementation refuses to apply a design costing more
than 5%.

The contrast is clean within charge strata. Comparing `hi_hi_hi` against
`hi_lo_hi`, for instance, holds charge at 8.99 and moment high while varying only
face fragmentation. One limitation should be noted: the high-charge stratum is
pinned near the top of the design window (8.99 across all four cells) rather than
spanning a range, so the charge factor is coarse.

With twenty-five peptides drawn from fifty, roughly three per cell will be
measured. That is a design capable of detecting a large effect and not a subtle
one, which is the appropriate expectation given four null results indicating the
effect is at most small.

## Output diversity

| Statistic | Top 100 | Draw pool (top 50) |
|---|---|---|
| Mean pairwise identity | 0.396 | 0.378 |
| Maximum pairwise identity | 0.963 | 0.967 |
| Mean length | 18.9 | 25.6 |
| Length range | 12-30 | 14-30 |

The maximum reflects one near-duplicate pair. This violates no rule — the 80%
ceiling applies to reference peptides, not within the submitted set — but it
consumes a slot that could have carried independent information.

## Predictive model performance

Held-out, clustered splits at 40% identity, five repeats.

| Head | n | Spearman | Within 1 dilution |
|---|---|---|---|
| Gram-negative MIC | 11,275 | 0.540 | 0.399 |
| HC50 | 5,550 | 0.482 | 0.356 |
| Gram-positive MIC | 10,845 | 0.462 | 0.375 |
| Resistant MIC | 2,016 | 0.391 | 0.406 |

These numbers are lower than commonly reported for this task. The
evaluation-protocol ablation quantifies why: +0.072 Spearman of the difference is
attributable to random splitting and to treating assay ceilings as measurements.

Within-one-dilution accuracy near 0.4 means typical errors are three- to
four-fold in concentration. The models are adequate for ranking and inadequate
for trusting any individual predicted value, which is precisely why the tolerance
band exists and why the draw pool is a designed experiment rather than a
confident top-fifty.
