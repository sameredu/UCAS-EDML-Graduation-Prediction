# Target leakage in graduation prediction from administrative records

Reference implementation for:

> Yaghi, S., & Baraka, R. S. *Target Leakage in Machine Learning Models that Predict Higher Education Graduation from Administrative Records.* (under review)

Twenty years of institutional records from the University College of Applied Sciences (UCAS) in Gaza, Palestine — 4,704 students across four IT specialisations, 2000–2019.

---

## What this repository shows

Two attributes held in almost every student information system are **deterministic functions of the graduation outcome**:

| Attribute | Why it leaks |
|---|---|
| `GRADUATE_YEAR` | Recorded only for students who graduated — null for every non-graduate |
| `STD_GRADEAVG` | Equals `STD_AVG` for every non-graduate; differs only once a student graduates |

Keep them, and a depth-limited decision tree, a bagged ensemble and a boosted ensemble all reach **exactly 100% accuracy with a cross-validation standard deviation of zero**. Drop them, and the same three models land in the high eighties.

```
CONFIGURATION A: with leakage attributes
  Random Forest    100.00   AUC 1.000   5-fold CV 100.00 +/- 0.00
  Decision Tree    100.00   AUC 1.000   5-fold CV 100.00 +/- 0.00
  GBT Classifier   100.00   AUC 1.000   5-fold CV 100.00 +/- 0.00

CONFIGURATION B: corrected (full record)
  Random Forest     88.81   AUC 0.954   5-fold CV  88.67 +/- 0.58
  Decision Tree     87.18   AUC 0.933   5-fold CV  87.29 +/- 0.54
  GBT Classifier    89.16   AUC 0.955   5-fold CV  88.50 +/- 0.62

CONFIGURATION C: pre-enrolment attributes only
  Random Forest     67.92   AUC 0.747   5-fold CV  68.30 +/- 1.30
  Decision Tree     67.28   AUC 0.738   5-fold CV  68.13 +/- 0.77
  GBT Classifier    70.47   AUC 0.769   5-fold CV  70.39 +/- 0.66
```

**The diagnostic worth taking away:** near-perfect accuracy across structurally different model families, combined with negligible cross-validation variance, is a signature of leakage rather than of a well-fitted model. Three learners that disagree about everything else do not agree perfectly unless one attribute is resolving the problem for all of them. `src/evaluation.py:leakage_check()` computes the arithmetic rather than asserting it.

---

## Relationship to the earlier version

This repository **supersedes** the implementation released with the first author's master's thesis, which reported 99.8% accuracy. That figure is the artefact diagnosed here. Two things were wrong in the earlier code:

1. **Ordering.** Normalisation and augmentation ran *before* the train/test split, so near-duplicate copies of training records appeared in the test partition.
2. **Attribute set.** `GRADUATE_YEAR` and `STD_GRADEAVG` were retained as model inputs.

Both are corrected here. `preprocessing.split_scale_augment()` is the only function that produces model-ready arrays and it enforces split → scale → augment; there is no code path that augments before splitting. The leakage attributes are materialised only for Configuration A, which exists to demonstrate the failure and is never presented as a result.

Results from the earlier version should not be cited as independent corroboration of the results here.

---

## Running it

```bash
pip install -r requirements.txt

python run_all.py --data data/sample_data.csv        # bundled 402-record sample
python run_all.py --data /path/to/full_extract.xls   # full dataset

python -m src.figures --data /path/to/full_extract.xls --out results/figures
```

`run_all.py` prints every table in the paper and writes `results/results.json`. With the full extract it reproduces the reported figures exactly; `--seed` is fixed at 42 throughout.

### About the bundled data

`data/sample_data.csv` is **synthetic**. The 400 records correspond to no real
student. We measured re-identification risk on an anonymised draw from the real
cohort first: on the full quasi-identifier set roughly three records in four were
unique, and on recorded sex plus specialisation plus exact GPA nearly nine in
ten. Publishing that would have let anyone who knew a former classmate's
enrolment year and major recover their exact GPA and graduation outcome, so the
sample is generated instead — see `data/README.md` and `src/make_sample.py`.

The synthetic sample preserves the structure this repository demonstrates: the
outcome mix, and the two leakage relationships exactly. Run `run_all.py` on it
and Configuration A still reaches 100% across all three models, which is the
claim you can verify for yourself without any real data.

It does **not** reproduce the paper's point estimates — 400 records give
confidence intervals three times wider. **The published results come from the
full dataset and are committed to `results/`**: `results_full.json`,
`results_full.txt` and the six figures in `results/figures/`.

---

## Layout

```
src/config.py          cohort filter, the three attribute configurations,
                       hyperparameters, seed — nothing derived from the outcome
src/preprocessing.py   cohort construction; split -> scale -> augment, in that order
src/models.py          the three classifiers
src/evaluation.py      Wilson intervals, stratified CV, McNemar, correlations,
                       the leakage check and the fairness audit
src/experiments.py     the experiments, mapped to the tables they produce
src/figures.py         the six figures
run_all.py             reproduces everything
src/make_sample.py     generates the synthetic sample
data/sample_data.csv   synthetic sample, illustrative only
results/               the published outputs, from the full dataset
```

---

## Attribute handling

**Modelled (Configuration B).** Age at enrolment, high-school branch, high-school score, IT major, region of residence, regular semesters completed, summer semesters completed, cumulative GPA.

**Excluded on leakage grounds.** `GRADUATE_YEAR`, `STD_GRADEAVG`.

**Audit only — never a model input.** Recorded sex. The source system stores a binary administrative field; it is not self-reported and does not represent gender identity. It is held out of every attribute set so that no prediction is conditioned on it, and used only to partition the test set for the subgroup audit. `split_scale_augment()` raises if it is passed as a feature.

Withholding it costs nothing: Random Forest accuracy *rises* from 88.31% to 88.81%, Gradient Boosting falls from 89.38% to 89.16% — both well inside the confidence intervals. The subgroup disparity persists anyway (demographic parity difference ≈ −0.11), which locates it in the enrolment-history attributes rather than in the model.

---

## Two findings that are easy to miss

**Augmentation hurts.** Gaussian-noise augmentation of the training set reduces accuracy for all three classifiers, by 0.71 to 1.91 points. It is reported as a negative control, not as a method. Replicating 3,292 records adds no information the originals do not already carry.

**Persistence beats attainment.** The number of regular semesters completed is the dominant attribute in all three models (RF 0.339, DT 0.628, GBT 0.570), ahead of cumulative GPA (0.246, 0.152, 0.180). The earlier ranking put GPA first — an appearance produced by the leaked graduate-GPA attribute.

---

## Known limitation in the data

Students still enrolled at extraction (21.83% of the cohort) are coded as non-graduates, which conflates *has not graduated yet* with *will not graduate*. The age-at-enrolment split that appears high in the decision tree is the visible trace of this censoring, and should not be read as evidence that older entrants perform better. A censoring-aware survival formulation would be preferable and is not implemented here.

---

## Data availability

`data/sample_data.csv` is synthetic and illustrative; it carries no information
about any individual. The full dataset is held by the UCAS Academic Affairs
Office and available under a data sharing agreement. Student identifiers were
replaced with anonymous IDs before any analysis, and the cohort is not
redistributable even so — `data/README.md` sets out the measured
re-identification risk that led to that decision.

The results and figures reported in the paper were produced from the full
dataset and are committed under `results/`.

## Licence

MIT — see `LICENSE`.
