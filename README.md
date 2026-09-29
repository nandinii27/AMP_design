# AMP Challenge 2027 submission

Generative design of antimicrobial peptides under multi-objective Chebyshev
scalarisation, with homology-controlled evaluation.

- **Abstract** — `ABSTRACT.md`
- **Training data disclosure** — `TRAINING_DATA.md`
- **Selection and ranking procedure** — `RANKING.md`

## Reproducing the submission

```bash
uv sync
uv run generate
```

Writes `generate/library.fasta` (50,000 sequences), `generate/top.fasta`
(100 ranked candidates), `generate/top_ranking.csv` (per-candidate breakdown) and
`generate/run_summary.json`.

Model weights and the parsed activity table are committed, so a fresh clone
reproduces the submission without retraining or re-downloading. Runtime is
roughly 40 minutes on four CPU threads. Verified with the organisers'
`verify_submission.py`: all checks passed.

## Method

A character-level transformer language model (809k parameters, 20 epochs,
trained on the 29,023 cysteine-free members of `data/antibacterial.fasta`)
proposes candidates. Four gradient-boosted regressors predict log2 MIC against
Gram-positive and Gram-negative panels, log2 HC50 on human erythrocytes, and
log2 MIC against resistant clinical isolates. Candidates are scored by augmented
Chebyshev scalarisation, refined by register-aware simulated annealing, screened
for novelty, and selected for diversity.

Held-out performance, clustered splits at 40% identity, five repeats:

| Head | n | Spearman | Within 1 dilution |
|---|---|---|---|
| Gram-negative MIC | 11,275 | 0.540 | 0.399 |
| HC50 | 5,550 | 0.482 | 0.356 |
| Gram-positive MIC | 10,845 | 0.462 | 0.375 |
| Resistant MIC | 2,016 | 0.391 | 0.406 |

## Findings

Under homology-controlled evaluation, no family of sequence-derived structural
descriptor improved prediction over amino acid composition. Tested and not
supported: hydrophobic moment and spectral purity, circular helical-wheel
statistics, H0 persistence of the three-dimensional hydrophobic point cloud, and
a conditional-disorder block modelling the coil-to-helix transition. Replicated
across two endpoints. Details and error bars in `ABSTRACT.md`; raw results in
`checkpoint/ablation*.json` and `checkpoint/disorder_*.json`.

An evaluation-protocol ablation quantifies inflation from random splitting and
from recording assay ceilings as measurements at +0.072 Spearman, of which the
ceiling effect (+0.051) is larger than the leakage effect (+0.021).

## Repository layout

```
data/          reference set and the parsed activity table
checkpoint/    trained weights, predictors, ablation results
src/amp_design/
  generate.py    entry point
  lm.py          language model
  features.py    physicochemical and arrangement descriptors
  topology.py    H0 persistence of the hydrophobic point cloud
  disorder.py    conditional-disorder descriptors
  models.py      censored regressors, in-distribution gate
  scalarize.py   Chebyshev objective
  search.py      register-aware annealing
  select.py      diverse subset selection
  design.py      factorial selection (exploratory, not applied)
  funnel.py      screening cascade
scripts/
  fetch_dbaasp.py        data retrieval
  parse_dbaasp.py        unit normalisation, censoring, filtering
  train.py               language model and predictors
  ablate.py              descriptor tier ablation
  ablate_protocol.py     evaluation protocol ablation
  ablate_disorder.py     conditional-disorder ablation
  design_top.py          factorial design report
```
<img width="527" height="370" alt="image" src="https://github.com/user-attachments/assets/6914956f-7b39-4694-8160-2e1da2860811" />

## Licence

MIT.
