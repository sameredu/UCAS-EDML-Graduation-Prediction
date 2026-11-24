"""
Evaluation: point metrics with Wilson score intervals, stratified
cross-validation, McNemar's test, Pearson correlations and the fairness audit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (confint_proportions_2indep,
                                          proportion_confint)

from . import config as C
from . import models


def evaluate(y_true, y_pred, y_prob) -> dict:
    """Point metrics for one fitted model on one test partition."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    lo, hi = proportion_confint(int(tp + tn), n, alpha=0.05, method="wilson")
    return {
        "CA": accuracy_score(y_true, y_pred) * 100,
        "CI_low": lo * 100,
        "CI_high": hi * 100,
        "AUC": roc_auc_score(y_true, y_prob),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "FNR": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
        # Advising view: an alert goes to every student predicted NOT to graduate.
        "alerts": int(tn + fn),
        "missed_alerts": int(fp),        # did not graduate, predicted to graduate
        "unneeded_alerts": int(fn),      # graduated, predicted not to
        "majority_baseline": float(max(np.mean(y_true), 1 - np.mean(y_true)) * 100),
    }


def evaluate_multiclass(y_true, y_pred, labels=(0, 1, 2)) -> dict:
    """Three outcomes at extraction: left, graduated, still enrolled."""
    from sklearn.metrics import recall_score as _rec
    rec = _rec(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    n = len(y_true)
    correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")
    return {
        "CA": correct / n * 100, "CI_low": lo * 100, "CI_high": hi * 100,
        "macro_F1": float(f1_score(y_true, y_pred, labels=list(labels),
                                   average="macro", zero_division=0)),
        "recall_left": float(rec[0]), "recall_graduated": float(rec[1]),
        "recall_still_enrolled": float(rec[2]),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(labels)).tolist(),
        "majority_baseline": float(np.bincount(np.asarray(y_true)).max() / n * 100),
    }


def cross_validate(df: pd.DataFrame, features: list[str],
                   folds: int = C.CV_FOLDS, seed: int = C.SEED,
                   target: str = C.TARGET) -> dict:
    """
    Stratified k-fold on the ORIGINAL records, without augmentation.
    Imputation and the scaler are refitted inside every fold.

    Fold scores share training records and are not independent, so no
    confidence interval is computed from them (v2.0.0 reported one); the five
    scores, their mean and their standard deviation are returned.
    """
    from .preprocessing import impute
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    scores = {name: [] for name in C.MODEL_ORDER}
    for tr, te in skf.split(X, y):
        A, B = impute(X[tr], X[te], features)
        scaler = MinMaxScaler().fit(A)
        A, B = scaler.transform(A), scaler.transform(B)
        for name, model in models.build_all(seed).items():
            model.fit(A, y[tr])
            scores[name].append(accuracy_score(y[te], model.predict(B)) * 100)

    out = {}
    for name, v in scores.items():
        v = np.asarray(v)
        out[name] = {
            "folds": np.round(v, 2).tolist(),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
        }
    return out


def mcnemar_tests(y_true, predictions: dict) -> dict:
    """Pairwise McNemar with continuity correction, on the shared test set."""
    names = list(predictions)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = predictions[names[i]] == y_true
            b = predictions[names[j]] == y_true
            b01, b10 = int((a & ~b).sum()), int((~a & b).sum())
            table = [[int((a & b).sum()), b01],
                     [b10, int((~a & ~b).sum())]]
            if b01 + b10 == 0:
                # No discordant pairs: the two models agree on every test case,
                # which is what Configuration A produces. The test is undefined
                # rather than infinitely significant.
                out[f"{names[i]} vs {names[j]}"] = {
                    "table": table, "chi2": None, "p": None,
                    "note": "undefined: no discordant predictions",
                }
                continue
            res = mcnemar(table, exact=False, correction=True)
            out[f"{names[i]} vs {names[j]}"] = {
                "table": table,
                "chi2": float(res.statistic),
                "p": float(res.pvalue),
            }
    # Holm adjustment across the pairwise comparisons in this configuration.
    keys = [k for k, v in out.items() if v["p"] is not None]
    if keys:
        _, p_adj, _, _ = multipletests([out[k]["p"] for k in keys], method="holm")
        for k, pa in zip(keys, p_adj):
            out[k]["p_holm"] = float(pa)
            out[k]["significant_holm"] = bool(pa < 0.05)
    return out


def correlations(df: pd.DataFrame, attributes: list[str]) -> dict:
    """Pearson r between each attribute and the binary outcome."""
    out = {}
    for a in attributes:
        sub = df[[a, C.TARGET]].dropna()
        r, p = pearsonr(sub[a].to_numpy(dtype=float), sub[C.TARGET].to_numpy(dtype=float))
        out[a] = {"r": float(r), "p": float(p), "n": int(len(sub))}
    return out


def leakage_check(df: pd.DataFrame) -> dict:
    """
    The arithmetic behind the leakage claim, computed rather than asserted.

    If `graduate_year_matches_label` is True, a rule of the form
    'a graduation year is recorded' reproduces the label exactly.
    """
    grad = (df[C.TARGET] == 1).to_numpy()
    same = np.isclose(df["STD_GRADEAVG"].to_numpy(dtype=float),
                      df["STD_AVG"].to_numpy(dtype=float),
                      rtol=C.GPA_RTOL, atol=C.GPA_ATOL)
    return {
        "n_graduated": int(grad.sum()),
        "n_not_graduated": int((~grad).sum()),
        "graduate_year_present_among_graduates":
            int(df.loc[grad, "GRADUATE_YEAR"].notna().sum()),
        "graduate_year_present_among_non_graduates":
            int(df.loc[~grad, "GRADUATE_YEAR"].notna().sum()),
        "graduate_year_matches_label":
            bool((df["GRADUATE_YEAR"].notna().astype(int) == df[C.TARGET]).all()),
        "gradeavg_equals_avg_among_non_graduates":
            float(same[~grad].mean()),
        "gradeavg_equals_avg_among_graduates":
            float(same[grad].mean()),
        # Counts (Reviewer 1, point 3): a DIFFERING value occurs only among
        # graduates; a matching value occurs in both groups.
        "n_graduates_matching": int(same[grad].sum()),
        "n_graduates_differing": int((~same[grad]).sum()),
        "n_non_graduates_matching": int(same[~grad].sum()),
        "n_non_graduates_differing": int((~same[~grad]).sum()),
        "tolerance": {"rtol": C.GPA_RTOL, "atol": C.GPA_ATOL},
    }


def _diff_ci(k1, n1, k2, n2) -> list:
    """95% Newcombe (hybrid score) interval for p1 - p2, independent groups."""
    if min(n1, n2) == 0:
        return [float("nan"), float("nan")]
    lo, hi = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb", compare="diff")
    return [float(lo), float(hi)]


def fairness_audit(y_true, y_pred, group) -> dict:
    """
    Subgroup counts and error rates by an attribute the model never saw.

    v2.1.0: called with the predictions of the UN-AUGMENTED models (the ones in
    Fig. 4); records with unknown sex (group 0) are excluded and counted.
    Selection rate = share predicted to have graduated. Differences are male
    minus female, with 95% Newcombe intervals.
    """
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    rows = {"excluded_unknown": int((group == 0).sum())}
    raw = {}
    for value, label in [(1, "male"), (2, "female")]:
        k = group == value
        tn, fp, fn, tp = confusion_matrix(y_true[k], y_pred[k], labels=[0, 1]).ravel()
        raw[label] = dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), n=int(k.sum()))
        rows[label] = {
            "n": int(k.sum()),
            "correct": int(tp + tn), "incorrect": int(fp + fn),
            "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "accuracy": float((tp + tn) / k.sum() * 100) if k.sum() else float("nan"),
            "TPR": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "FNR": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
            "FPR": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "selection_rate": float(y_pred[k].mean()) if k.sum() else float("nan"),
            "base_rate": float(y_true[k].mean()) if k.sum() else float("nan"),
            # advising view
            "alerts": int(tn + fn),
            "missed_alerts": int(fp),
            "non_graduates": int(fp + tn),
        }
    m, f = raw["male"], raw["female"]
    rows["demographic_parity_difference"] = (
        rows["male"]["selection_rate"] - rows["female"]["selection_rate"])
    rows["demographic_parity_ci"] = _diff_ci(m["tp"] + m["fp"], m["n"], f["tp"] + f["fp"], f["n"])
    rows["equal_opportunity_difference"] = rows["male"]["TPR"] - rows["female"]["TPR"]
    rows["equal_opportunity_ci"] = _diff_ci(m["tp"], m["tp"] + m["fn"], f["tp"], f["tp"] + f["fn"])
    rows["fpr_difference"] = rows["male"]["FPR"] - rows["female"]["FPR"]
    rows["fpr_ci"] = _diff_ci(m["fp"], m["fp"] + m["tn"], f["fp"], f["fp"] + f["tn"])
    rows["accuracy_difference"] = (rows["male"]["accuracy"] - rows["female"]["accuracy"]) / 100
    rows["accuracy_ci"] = _diff_ci(m["tp"] + m["tn"], m["n"], f["tp"] + f["tn"], f["n"])
    rows["base_rate_difference"] = rows["male"]["base_rate"] - rows["female"]["base_rate"]
    rows["base_rate_ci"] = _diff_ci(m["tp"] + m["fn"], m["n"], f["tp"] + f["fn"], f["n"])
    return rows


def concordance_index(time, event, risk) -> float:
    """Harrell's C: higher `risk` should mean an earlier event."""
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    for i in np.where(event == 1)[0]:
        comparable = time > time[i]
        den += comparable.sum()
        num += (risk[i] > risk[comparable]).sum() + 0.5 * (risk[i] == risk[comparable]).sum()
    return float(num / den) if den else float("nan")
