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
from statsmodels.stats.proportion import proportion_confint

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
    }


def cross_validate(df: pd.DataFrame, features: list[str],
                   folds: int = C.CV_FOLDS, seed: int = C.SEED) -> dict:
    """
    Stratified k-fold on the ORIGINAL records, without augmentation.
    The scaler is refitted inside every fold.
    """
    X = df[features].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    scores = {name: [] for name in C.MODEL_ORDER}
    for tr, te in skf.split(X, y):
        scaler = MinMaxScaler().fit(X[tr])
        A, B = scaler.transform(X[tr]), scaler.transform(X[te])
        for name, model in models.build_all(seed).items():
            model.fit(A, y[tr])
            scores[name].append(accuracy_score(y[te], model.predict(B)) * 100)

    out = {}
    for name, v in scores.items():
        v = np.asarray(v)
        sd = v.std(ddof=1)
        half = 1.96 * sd / np.sqrt(folds)
        out[name] = {
            "folds": np.round(v, 2).tolist(),
            "mean": float(v.mean()),
            "sd": float(sd),
            "ci": [float(v.mean() - half), float(v.mean() + half)],
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
    return out


def correlations(df: pd.DataFrame, attributes: list[str]) -> dict:
    """Pearson r between each attribute and the binary outcome."""
    y = df[C.TARGET].to_numpy(dtype=float)
    out = {}
    for a in attributes:
        r, p = pearsonr(df[a].to_numpy(dtype=float), y)
        out[a] = {"r": float(r), "p": float(p)}
    return out


def leakage_check(df: pd.DataFrame) -> dict:
    """
    The arithmetic behind the leakage claim, computed rather than asserted.

    If `graduate_year_matches_label` is True, a rule of the form
    'a graduation year is recorded' reproduces the label exactly.
    """
    grad = df[C.TARGET] == 1
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
            float(np.isclose(df.loc[~grad, "STD_GRADEAVG"],
                             df.loc[~grad, "STD_AVG"]).mean()),
        "gradeavg_equals_avg_among_graduates":
            float(np.isclose(df.loc[grad, "STD_GRADEAVG"],
                             df.loc[grad, "STD_AVG"]).mean()),
    }


def fairness_audit(y_true, y_pred, group) -> dict:
    """
    Subgroup error rates by an attribute the model never saw.

    Positive demographic-parity and equal-opportunity differences favour the
    first group; the paper reports male minus female.
    """
    rows = {}
    for value, label in [(1, "male"), (2, "female")]:
        k = group == value
        tn, fp, fn, tp = confusion_matrix(y_true[k], y_pred[k], labels=[0, 1]).ravel()
        rows[label] = {
            "n": int(k.sum()),
            "accuracy": accuracy_score(y_true[k], y_pred[k]) * 100,
            "TPR": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
            "FNR": float(fn / (tp + fn)) if (tp + fn) else float("nan"),
            "FPR": float(fp / (fp + tn)) if (fp + tn) else float("nan"),
            "selection_rate": float(y_pred[k].mean()),
            "base_rate": float(y_true[k].mean()),
        }
    rows["demographic_parity_difference"] = (
        rows["male"]["selection_rate"] - rows["female"]["selection_rate"])
    rows["equal_opportunity_difference"] = rows["male"]["TPR"] - rows["female"]["TPR"]
    rows["base_rate_difference"] = rows["male"]["base_rate"] - rows["female"]["base_rate"]
    return rows
