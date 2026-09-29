# Training data statement

Prepared for the AMP Challenge 2027 full-requirements submission. All sources are
public; no proprietary or non-public data was used.

## Sources

| Source | Retrieved | Role | In repository |
|---|---|---|---|
| `data/antibacterial.fasta` | provided with the challenge starter repository | generator training corpus; novelty reference | yes |
| DBAASP v3, REST API at `https://dbaasp.org` | 29 September 2026 | activity labels (MIC, HC50) | parsed table only |

Nothing else was used. APD and Peptipedia were evaluated and not used: measured
overlap between DBAASP monomers and the challenge reference set is 92.2%
(8,614 of 9,346 usable monomers in a partial sample), and both alternatives
aggregate from the same primary literature, so neither adds sequence coverage
beyond the reference set nor corrects the label artifacts described below.

## Generator corpus

The language model was trained on `data/antibacterial.fasta` only.

- 39,448 sequences supplied, all unique, all within the challenge alphabet and
  the 8-50 length range (median 18, maximum exactly 50) as distributed.
- 10,425 contain cysteine (26.4%). These were excluded, leaving **29,023
  sequences**, because the design forbids cysteine: a free thiol dimerises and
  the challenge prohibits the modifications that would cap it. Training on
  sequences whose defining residue the generator can never emit spends model
  capacity on an unreachable region.
- No other filtering. No synthetic or augmented sequences.

The reference set is simultaneously the training corpus and the novelty
constraint, since the top-100 must remain below 80% Levenshtein identity to every
member. This is a deliberate choice and not an oversight: with a median length of
18 the accessible space is approximately 2.6 x 10^23 sequences, so 39,448
exclusion balls of radius roughly four edits occupy a negligible fraction of it.
The constraint binds only if the generator memorises, which sampling temperature
and the novelty filter control. The empirical rejection rate was low: 3,995 of
4,000 scored candidates passed the identity screen.

## Activity labels

Retrieved from the DBAASP REST API on 29 September 2026. The search endpoint
(`GET /peptides`) returns flattened summaries without per-species activity, so
peptide identifiers were paginated from it and full records fetched individually
from `GET /peptides/{id}`. The search index reported 25,542 peptides; all were
retrieved.

### Filters applied, in order

1. **Monomers only.** Multimeric and multi-peptide entries dropped (675 records).
   The challenge requires linear single-chain peptides.
2. **Canonical residues only.** Records flagged with unusual amino acids dropped
   (4,993 records).
3. **Alphabet and length.** Sequences outside `ACDEFGHIKLMNPQRSTVWY` or outside
   8-50 residues dropped (1,988 records).
4. **MIC-type measures only.** `activityMeasureGroup` restricted to MIC, MIC50
   and MIC90; 38,206 activity rows dropped. Excluded types were MBC (bactericidal
   rather than inhibitory, and systematically higher), IC50, EC50, LC50, LD50,
   MEC and MFC (minimum fungicidal concentration). Pooling these with MIC would
   mix distinct physical quantities.
5. **Bacterial targets only.** 14,012 activity rows dropped, including all
   *Candida* and viral records. Antifungal and antiviral activity operate on
   different membrane chemistry.
6. **Human erythrocytes only** for haemolysis. Rabbit, horse, sheep, rat and
   mouse erythrocyte records dropped, along with non-erythrocyte cell lines
   (5,334 rows). Human records are 77% of the haemolysis data, and erythrocyte
   sensitivity varies substantially between species.

After filtering: **17,886 usable peptides**, 87,311 MIC rows and 11,523
haemolysis rows.

### Unit normalisation

Roughly 44% of activity values are reported in ug/mL and the remainder in uM,
while the competition's potency threshold is 16 uM. Conversion uses the
monoisotopic molecular weight of each peptide, which varies approximately
twofold across the legal length range; a mixed-unit table would corrupt the
target by an unpredictable per-peptide factor. Records whose unit could not be
resolved were dropped (20 rows). All stored values are uM.

### Censoring treatment

Two distinct forms, both handled explicitly rather than by imputation.

**Explicit bounds.** Values written as ">100", ">128" and similar are recorded as
right-censored at the stated bound.

**Implicit ceilings — a stated assumption.** The reported haemolysis distribution
concentrates at round values (100, 50, 200, 250, 128, 256, 512 uM), and "100"
appears 1,244 times as a bare number against only 254 times as ">100". The most
plausible reading is that most authors who tested to the top of their dilution
series and observed no lysis recorded the ceiling without the inequality. Such
rows describe the experiment's budget rather than a property of the molecule.

**I therefore treat a bare value falling exactly on a standard assay ceiling as
right-censored.** This is a population-level inference from the value
distribution, not a determination about any individual record, and it is stated
here because it is a judgement call that materially affects the HC50 model. It
errs in the conservative direction: mislabelling a true measurement as censored
loses information, whereas mislabelling a ceiling as a measurement injects noise
into the target. The correction reclassified a substantial fraction of the
haemolysis rows and raised final HC50 coverage to 5,550 sequences of which 62%
are censored; held-out performance improved despite the reduced effective sample
size.

**Intervals.** Ranged values such as "4-8" reflect the two-fold dilution series
reporting its own resolution. These are collapsed to the upper bound, which is
the lowest well showing inhibition, and flagged as interval-censored.

### Aggregation

Replicate measurements are collapsed per sequence and endpoint group using the
median on the log2 scale, the natural scale for a two-fold dilution series. A
sequence is marked right-censored only when every contributing measurement was.

### Endpoint groups

| Group | Sequences | Right-censored |
|---|---|---|
| Gram-positive MIC | 10,845 | 1,997 |
| Gram-negative MIC | 11,275 | 1,786 |
| HC50, human erythrocytes | 5,550 | 3,454 |
| MIC, resistant isolates | 2,016 | 512 |

The resistant group is assembled from strains carrying a documented resistance
phenotype (MRSA, ESBL and vancomycin-resistant markers, including
*S. aureus* ATCC 43300 and *K. pneumoniae* ATCC 700603). It is underpowered
relative to the others and its held-out Spearman is correspondingly lower
(0.391 against 0.462 and 0.540). It contributes a weighted shift to the potency
axes rather than forming an objective of its own, and no independent
multidrug-resistance model is claimed.

## Evaluation protocol

All reported performance uses **greedy single-linkage clustering at 40% sequence
identity**, with whole clusters assigned to train or test. Peptide databases are
saturated with analogue series differing at a single position; a random split
places siblings on both sides and inflates every metric. The quantified effect is
reported in the abstract and in `checkpoint/protocol_gram_negative.json`.

## Manual intervention

None beyond the filters above. No candidate sequences were hand-selected, hand-
edited or removed. The top-100 is produced entirely by the scored pipeline and
the factorial selection described in the ranking documentation.

## Reproducibility

- Dependency management via `uv`; `uv.lock` committed.
- Python version pinned in `.python-version`.
- All randomness seeded from a single `--seed` (default 42), covering Python,
  NumPy and PyTorch, with deterministic algorithms enabled.
- Model weights and the parsed activity table are committed, so a fresh clone
  reproduces the submission without retraining or re-downloading.
- The raw DBAASP dump (223 MB) is excluded from the repository for size reasons;
  `scripts/fetch_dbaasp.py` reproduces it, and `scripts/parse_dbaasp.py`
  regenerates `data/activity_wide.csv` from it deterministically.
- Verified with the organisers' `scripts/verify_submission.py`, which clones the
  repository, installs dependencies, generates twice and compares byte for byte.
  All checks passed.

## Licence

MIT. All derived data in this repository is released under the same terms.
