#!/usr/bin/env python3
"""
Reproduce every number reported in the paper.

    python run_all.py --data data/sample_data.csv       # bundled sample
    python run_all.py --data /path/to/full_extract.xls  # full dataset

Writes results/results.json and prints the tables to stdout.
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


def report(R: dict) -> None:
    cfg = R["configurations"]
    d = R["descriptives"]

    section("COHORT")
    print(f"  analytical cohort                      {R['cohort_size']:>8,}")
    print(f"  train / test                           "
          f"{cfg['B']['n_train']:>8,} / {cfg['B']['n_test']:,}")
    print(f"  graduated                              {d['graduated']:>8,}"
          f"   ({d['graduation_rate']:.2f}%)")
    print(f"  recorded sex (audit only)              "
          f"{d['male_pct']:.2f}% male / {d['female_pct']:.2f}% female")

    section("LEAKAGE DIAGNOSTIC   (Section 5.2)")
    L = R["leakage_check"]
    print(f"  graduation year recorded, graduates        "
          f"{L['graduate_year_present_among_graduates']:>7,} / {L['n_graduated']:,}")
    print(f"  graduation year recorded, non-graduates    "
          f"{L['graduate_year_present_among_non_graduates']:>7,} / {L['n_not_graduated']:,}")
    print(f"  reproduces the label exactly               "
          f"{str(L['graduate_year_matches_label']):>7}")
    print(f"  graduate GPA equals cumulative GPA         "
          f"{L['gradeavg_equals_avg_among_non_graduates'] * 100:>6.1f}% of non-graduates")

    for name in ["A", "B", "C"]:
        block = cfg[name]
        section(f"CONFIGURATION {C.CONFIG_LABELS[name]}")
        header = (f"  {'model':<16}{'CA %':>7}{'95% CI':>18}{'AUC':>7}{'F1':>7}"
                  f"{'Recall':>8}{'FN':>6}{'aug x4':>9}{'5-fold CV':>18}")
        print(header)
        for m in C.MODEL_ORDER:
            r = block["no_augmentation"][m]
            aug = block["with_augmentation"][m]["CA"]
            cv = block["cross_validation"][m]
            ci = f"[{r['CI_low']:.1f}, {r['CI_high']:.1f}]"
            cvs = f"{cv['mean']:.2f} +/- {cv['sd']:.2f}"
            print(f"  {m:<16}{r['CA']:>7.2f}{ci:>18}{r['AUC']:>7.3f}{r['F1']:>7.3f}"
                  f"{r['Recall']:>8.3f}{r['FN']:>6}{aug:>9.2f}{cvs:>18}")
        print("  McNemar:")
        for pair, t in block["mcnemar"].items():
            if t["p"] is None:
                print(f"    {pair:<36} {t['note']}")
                continue
            verdict = "significant" if t["p"] < 0.05 else "not significant"
            print(f"    {pair:<36} chi2 = {t['chi2']:>7.3f}   "
                  f"p = {fmt_p(t['p']):<9} {verdict}")

    section("FEATURE IMPORTANCE   (Configuration B)")
    imp = {m: cfg["B"]["no_augmentation"][m]["importances"] for m in C.MODEL_ORDER}
    print(f"  {'attribute':<26}" + "".join(f"{m:>17}" for m in C.MODEL_ORDER))
    for f in sorted(imp["Random Forest"], key=imp["Random Forest"].get, reverse=True):
        print(f"  {C.DISPLAY_NAMES.get(f, f):<26}"
              + "".join(f"{imp[m][f]:>17.3f}" for m in C.MODEL_ORDER))
    dt = cfg["B"]["no_augmentation"]["Decision Tree"]
    print(f"\n  decision tree: {dt['n_nodes']} nodes, {dt['n_leaves']} leaves")

    section("CORRELATION WITH GRADUATION   (Table 2)")
    for f, v in sorted(R["correlations"].items(), key=lambda kv: -abs(kv[1]["r"])):
        if f in C.LEAKAGE_ATTRIBUTES:
            flag = "   <- excluded (leakage)"
        elif f in C.AUDIT_ONLY:
            flag = "   <- audit only"
        else:
            flag = ""
        print(f"  {C.DISPLAY_NAMES.get(f, f):<26}r = {v['r']:+.4f}   "
              f"p = {fmt_p(v['p']):<9}{flag}")

    section("FAIRNESS AUDIT BY RECORDED SEX   (Section 5.9)")
    print("  recorded sex is NOT an input to any model; it partitions the test set only\n")
    print(f"  {'model':<16}{'group':<9}{'n':>6}{'acc %':>8}{'TPR':>8}{'FNR':>8}"
          f"{'FPR':>8}{'select':>9}{'base':>8}")
    for m in C.MODEL_ORDER:
        v = R["fairness"][m]
        for g in ["male", "female"]:
            x = v[g]
            label = m if g == "male" else ""
            print(f"  {label:<16}{g:<9}{x['n']:>6}{x['accuracy']:>8.2f}"
                  f"{x['TPR']:>8.3f}{x['FNR']:>8.3f}{x['FPR']:>8.3f}"
                  f"{x['selection_rate']:>9.3f}{x['base_rate']:>8.3f}")
        print(f"  {'':<16}{'M - F':<9}{'':>6}{'':>8}"
              f"{v['equal_opportunity_difference']:>8.3f}{'':>8}{'':>8}"
              f"{v['demographic_parity_difference']:>9.3f}"
              f"{v['base_rate_difference']:>8.3f}")
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
