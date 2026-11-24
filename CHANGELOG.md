# Changelog

## 2.1.0 — revision in response to peer review (Discover Artificial Intelligence)

Fixes
- **Fairness audit uses the un-augmented models.** v2.0.0 called the audit on
  the models trained with fourfold augmentation, so Table 7 pooled to the
  augmented accuracies (e.g. 1,227/1,412 = 86.90% for Random Forest) rather
  than the 88.81% of Fig. 4. The audit now uses the same predictions as Fig. 4,
  reports correct/incorrect counts per subgroup and 95% Newcombe intervals for
  the male − female differences, and prints a pooled-count check.
- **Recorded sex**: missing or unrecognised values are coded 0 and excluded
  from the audit (previously assigned to male). Counts are reported.
- **Age imputation** is fitted on the training partition only, and on each
  training fold inside cross-validation (previously the whole-cohort median,
  before the split). Counts are reported.
- **McNemar**: Holm-adjusted p-values added; continuity correction stated.
- **Cross-validation**: the confidence interval computed from fold scores is
  removed (folds share training records); fold scores, mean and SD are reported.
- **Leakage check** reports counts: graduates and non-graduates whose graduate
  GPA matches the cumulative GPA, with the tolerance used.
- **Synthetic sample**: about 55% of synthetic graduates now have a matching
  graduate GPA, as in the real data; still-enrolled students are concentrated in
  recent cohorts; a few missing ages and sex values exercise the new rules.

Additions
- Nine-algorithm screening, five-fold CV on the training partition only.
- Advising view for every model: alerts, missed alerts, unneeded alerts,
  majority-class baseline, training accuracy (Decision Tree is a single tree).
- Configuration B-S (B without semester counts).
- Outcome mix by admission year and follow-up time (extract taken in mid-2020;
  admission years up to 2000 are reported in one row, since that field holds
  some data-entry errors and is not a model input).
- Sensitivity analyses: still-enrolled students excluded; three outcomes
  (left / graduated / still enrolled); complete-follow-up cohorts; temporal
  validation (earlier cohorts train, later cohorts test); complete follow-up =
  admitted up to 2014, at least six years before extraction; discrete-time hazard
  model of graduation on admission attributes with test C-index.

- `results/` regenerated with release 2.1.0 on the full extract (aggregate
  results only).

Not possible with this extract
- Landmark prediction at the end of semester k requires semester-level grades;
  the extract holds cumulative GPA and semester counts only.
