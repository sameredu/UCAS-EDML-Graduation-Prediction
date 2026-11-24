#!/usr/bin/env python3
"""
Reproduce every number reported in the paper.

    python run_all.py --data data/sample_data.csv       # bundled sample
    python run_all.py --data /path/to/full_extract.xls  # full dataset

Writes results/results.json and prints the tables to stdout.

v2.1.0: revised in response to peer review; see CHANGELOG.md.
"""
import argparse
import json
import os

from src import config as C
from src.experiments import run_all
from src.preprocessing import prepare

BAR = "=" * 86


def fmt_p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def section(title: str) -> None:
    print(f"\n{BAR}\n{title}\n{BAR}")


def _row(m, r):
    ci = f"[{r['CI_low']:.1f}, {r['CI_high']:.1f}]"
    return f"  {m:<16}{r['CA']:>7.2f}{ci:>16}{r['AUC']:>7.3f}{r['F1']:>7.3f}{r['Recall']:>8.3f}"


def _ci(v):
    return f"[{v[0]:+.3f}, {v[1]:+.3f}]"


def report(R: dict) -> None:
    cfg = R["configurations"]
    d = R["descriptives"]

    section("COHORT, MISSING VALUES AND FOLLOW-UP   (Section 3.1)")
    print(f"  analytical cohort                      {R['cohort_size']:>8,}")
    print(f"  train / test                           "
          f"{cfg['B']['n_train']:>8,} / {cfg['B']['n_test']:,}")
    print(f"  graduated                              {d['graduated']:>8,}"
          f"   ({d['graduation_rate']:.2f}%)")
    print(f"  still enrolled at extraction           {d['still_enrolled']:>8,}")
    print(f"  recorded sex (known values)            "
          f"{d['male_pct']:.2f}% male / {d['female_pct']:.2f}% female")
    M = R["missing"]
    print(f"  age missing (imputed, training median) {M['age_missing']:>8,}")
    print(f"  sex missing/unrecognised (excluded)    {M['sex_unknown']:>8,}")
    print(f"  region unknown (coded {C.REGION_UNKNOWN})               {M['region_unknown']:>8,}")
    F = R["followup"]
    print(f"\n  extraction date: {F['extraction_date'] or 'NOT SET (config.EXTRACTION_DATE)'}"
          f"; year used for follow-up: {F['extraction_year_used']}")
    print(f"  {'admitted':<16}{'n':>6}{'years':>7}{'grad %':>9}{'enrolled %':>12}{'left %':>9}")
    for y, v in F["by_admission_year"].items():
        print(f"  {str(y):<16}{v['n']:>6}{v['years_to_extraction']:>7}{v['graduated_pct']:>9.1f}"
              f"{v['still_enrolled_pct']:>12.1f}{v['left_pct']:>9.1f}")

    section("LEAKAGE DIAGNOSTIC   (Section 5.2)")
    L = R["leakage_check"]
    print(f"  graduation year recorded, graduates        "
          f"{L['graduate_year_present_among_graduates']:>7,} / {L['n_graduated']:,}")
    print(f"  graduation year recorded, non-graduates    "
          f"{L['graduate_year_present_among_non_graduates']:>7,} / {L['n_not_graduated']:,}")
    print(f"  reproduces the label exactly               "
          f"{str(L['graduate_year_matches_label']):>7}")
    print(f"  graduate GPA = cumulative GPA, non-grads   "
          f"{L['n_non_graduates_matching']:>7,} / {L['n_not_graduated']:,}")
    print(f"  graduate GPA = cumulative GPA, graduates   "
          f"{L['n_graduates_matching']:>7,} / {L['n_graduated']:,}"
          f"   ({L['gradeavg_equals_avg_among_graduates'] * 100:.2f}%)")
    print(f"  tolerance: numpy.isclose rtol={L['tolerance']['rtol']}, atol={L['tolerance']['atol']}")

    section("NINE-ALGORITHM SCREENING   (Section 3.4; training partition only, 5-fold CV)")
    Sc = R["screening"]
    print(f"  records: {Sc['n_records']:,} ({Sc['partition']})")
    for name, v in Sc.items():
        if isinstance(v, dict):
            print(f"  {name:<22}{v['accuracy_mean']:>7.2f} +/- {v['accuracy_sd']:.2f}   AUC {v['auc_mean']:.3f}")

    for name in ["A", "B", "B-S", "C"]:
        block = cfg[name]
        section(f"CONFIGURATION {C.CONFIG_LABELS[name]}")
        print(f"  {'model':<16}{'CA %':>7}{'95% CI':>16}{'AUC':>7}{'F1':>7}{'Recall':>8}"
              f"{'aug x4':>9}{'5-fold CV (folds; mean +/- SD)':>40}")
        for m in C.MODEL_ORDER:
            r = block["no_augmentation"][m]
            cv = block["cross_validation"][m]
            folds = " ".join(f"{x:.2f}" for x in cv["folds"])
            print(_row(m, r) + f"{block['with_augmentation'][m]['CA']:>9.2f}"
                  f"   {folds}; {cv['mean']:.2f} +/- {cv['sd']:.2f}")
        r0 = block["no_augmentation"][C.MODEL_ORDER[0]]
        print(f"  majority-class baseline: {r0['majority_baseline']:.2f}%")
        print("  advising view (alert = predicted not to graduate):")
        for m in C.MODEL_ORDER:
            r = block["no_augmentation"][m]
            print(f"    {m:<16} alerts {r['alerts']:>5}   missed (non-graduates, no alert) "
                  f"{r['missed_alerts']:>4} of {r['FP'] + r['TN']}   unneeded (graduates alerted) "
                  f"{r['unneeded_alerts']:>4} of {r['TP'] + r['FN']}   train acc {r['train_accuracy']:.2f}")
        print("  McNemar (continuity-corrected; Holm-adjusted across the three pairs):")
        for pair, t in block["mcnemar"].items():
            if t["p"] is None:
                print(f"    {pair:<36} {t['note']}")
                continue
            verdict = "significant" if t["significant_holm"] else "not significant"
            print(f"    {pair:<36} chi2 = {t['chi2']:>7.3f}   p = {fmt_p(t['p']):<9}"
                  f" p_Holm = {fmt_p(t['p_holm']):<9} {verdict}")

    section("FEATURE IMPORTANCE   (Configuration B)")
    imp = {m: cfg["B"]["no_augmentation"][m]["importances"] for m in C.MODEL_ORDER}
    print(f"  {'attribute':<26}" + "".join(f"{m:>17}" for m in C.MODEL_ORDER))
    for f in sorted(imp["Random Forest"], key=imp["Random Forest"].get, reverse=True):
        print(f"  {C.DISPLAY_NAMES.get(f, f):<26}"
              + "".join(f"{imp[m][f]:>17.3f}" for m in C.MODEL_ORDER))
    dt = cfg["B"]["no_augmentation"]["Decision Tree"]
    print(f"\n  decision tree (single CART tree): depth {dt['depth']}, {dt['n_nodes']} nodes, "
          f"{dt['n_leaves']} leaves; train {dt['train_accuracy']:.2f}% vs test {dt['CA']:.2f}%")

    section("CORRELATION WITH GRADUATION   (Table 2)")
    for f, v in sorted(R["correlations"].items(), key=lambda kv: -abs(kv[1]["r"])):
        if f in C.LEAKAGE_ATTRIBUTES:
            flag = "   <- excluded (leakage)"
        elif f in C.AUDIT_ONLY:
            flag = "   <- audit only"
        else:
            flag = ""
        print(f"  {C.DISPLAY_NAMES.get(f, f):<26}r = {v['r']:+.4f}   "
              f"p = {fmt_p(v['p']):<9} n = {v['n']:<6}{flag}")

    section("FAIRNESS AUDIT BY RECORDED SEX   (Section 5.9, Table 7)")
    print("  un-augmented Configuration B models (same predictions as Fig. 4);")
    print("  recorded sex is NOT an input; selection rate = share predicted to graduate")
    print(f"  pooled correct (check against Fig. 4): {R['fairness_check_pooled_correct']}\n")
    print(f"  {'model':<16}{'group':<8}{'n':>6}{'corr':>6}{'inc':>5}{'acc %':>8}{'TPR':>7}{'FNR':>7}"
          f"{'FPR':>7}{'select':>8}{'base':>7}{'missed':>8}")
    for m in C.MODEL_ORDER:
        v = R["fairness"][m]
        for g in ["male", "female"]:
            x = v[g]
            label = m if g == "male" else ""
            print(f"  {label:<16}{g:<8}{x['n']:>6}{x['correct']:>6}{x['incorrect']:>5}"
                  f"{x['accuracy']:>8.2f}{x['TPR']:>7.3f}{x['FNR']:>7.3f}{x['FPR']:>7.3f}"
                  f"{x['selection_rate']:>8.3f}{x['base_rate']:>7.3f}"
                  f"{x['missed_alerts']:>4}/{x['non_graduates']:<4}")
        print(f"  {'':<16}M - F   selection {v['demographic_parity_difference']:+.3f} {_ci(v['demographic_parity_ci'])}"
              f"   TPR {v['equal_opportunity_difference']:+.3f} {_ci(v['equal_opportunity_ci'])}"
              f"   FPR {v['fpr_difference']:+.3f} {_ci(v['fpr_ci'])}")
        print(f"  {'':<16}        accuracy {v['accuracy_difference']:+.3f} {_ci(v['accuracy_ci'])}"
              f"   base rate {v['base_rate_difference']:+.3f} {_ci(v['base_rate_ci'])}"
              f"   unknown sex excluded: {v['excluded_unknown']}")

    section("SENSITIVITY ANALYSES   (Section 5.11, Table 9)")
    S = R["sensitivity"]

    def block(title, B, cfgs):
        print(f"\n  {title}")
        for c in cfgs:
            if c not in B:
                continue
            b = B[c]
            print(f"    {C.CONFIG_LABELS[c]}  (train {b['n_train']:,} / test {b['n_test']:,})")
            for m in C.MODEL_ORDER:
                r = b["results"][m]
                if "AUC" in r:
                    print("  " + _row(m, r) + f"   baseline {r['majority_baseline']:.2f}")
                else:
                    print(f"      {m:<16}{r['CA']:>7.2f} [{r['CI_low']:.1f}, {r['CI_high']:.1f}]"
                          f"  macro-F1 {r['macro_F1']:.3f}  recall left/grad/enrolled "
                          f"{r['recall_left']:.3f}/{r['recall_graduated']:.3f}/{r['recall_still_enrolled']:.3f}"
                          f"  baseline {r['majority_baseline']:.2f}")

    block(f"(i) still-enrolled students excluded (n = {S['excluding_still_enrolled']['n']:,})",
          S["excluding_still_enrolled"], ["B", "B-S", "C"])
    block("(ii) three outcomes: left / graduated / still enrolled", S["three_outcomes"], ["B", "C"])
    cf = S["complete_followup"]
    block(f"(iii) complete follow-up: admitted up to {cf['admitted_up_to']} (n = {cf['n']:,}; "
          f"still enrolled in subset {cf['still_enrolled_in_subset']})", cf, ["B", "B-S", "C"])
    if "temporal_complete_followup" in S:
        T = S["temporal_complete_followup"]
        block(f"(iv) temporal validation within complete-follow-up cohorts: train <= "
              f"{T['train_admitted_up_to']}, test {T['test_admitted']}", T, ["B", "B-S", "C"])
    T = S["temporal_all_cohorts"]
    block(f"(v) temporal validation, all cohorts: train <= {T['train_admitted_up_to']}, "
          f"test {T['test_admitted']} ({T['note']})", T, ["B", "C"])
    sv = S["survival_admission_attributes"]
    print(f"\n  (vi) discrete-time hazard of graduation, admission attributes only")
    print(f"      train students {sv['n_train_students']:,}, person-periods {sv['n_person_periods']:,}, "
          f"events {sv['events_train']:,}; test C-index {sv['c_index_test']:.3f} "
          f"(n = {sv['n_test_students']:,})")
    for f, v in sv["odds_ratios_per_sd"].items():
        print(f"      {C.DISPLAY_NAMES.get(f, f):<24} OR per SD {v['OR']:.3f}  "
              f"[{v['CI'][0]:.3f}, {v['CI'][1]:.3f}]")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/sample_data.csv",
                    help="path to the extract (.csv, .xls/.xlsx or .pkl)")
    ap.add_argument("--out", default="results/results.json")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    df = prepare(args.data)
    R = run_all(df, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2, ensure_ascii=False, default=str)

    report(R)
    print(f"Full results written to {args.out}\n")


if __name__ == "__main__":
    main()
