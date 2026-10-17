"""
The experiments reported in the paper, each mapped to the table or figure it
produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import evaluation as ev
from . import models
from .preprocessing import split_scale_augment


def run_configuration(df: pd.DataFrame, name: str, augment_factor: int = 1,
                      seed: int = C.SEED) -> dict:
    """Fit all three models under one attribute configuration."""
    features = C.CONFIGURATIONS[name]
    X_tr, X_te, y_tr, y_te, test_idx = split_scale_augment(
        df, features, augment_factor=augment_factor, seed=seed)

    results, predictions, fitted = {}, {}, {}
    for model_name, model in models.build_all(seed).items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]

        row = ev.evaluate(y_te, y_pred, y_prob)
        row["importances"] = {
            f: float(round(v, 4))
            for f, v in zip(features, model.feature_importances_)
        }
        if model_name == "Decision Tree":
            row["n_nodes"] = int(model.tree_.node_count)
            row["n_leaves"] = int(model.get_n_leaves())

        results[model_name] = row
        predictions[model_name] = y_pred
        fitted[model_name] = model

    return {
        "features": features,
        "results": results,
        "predictions": predictions,
        "fitted": fitted,
        "y_test": y_te,
        "test_index": test_idx,
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }


def run_all(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Every number reported in Sections 3 to 5 of the paper."""
    out = {"cohort_size": int(len(df))}

    # --- Section 3.2, descriptive -------------------------------------------
    out["descriptives"] = {
        "graduated": int(df[C.TARGET].sum()),
        "graduation_rate": float(df[C.TARGET].mean() * 100),
        "male_pct": float((df["SEX_C"] == 1).mean() * 100),
        "female_pct": float((df["SEX_C"] == 2).mean() * 100),
        "grad_rate_by_branch":
            df.groupby("HS_BRANCH")[C.TARGET].mean().mul(100).round(2).to_dict(),
        "grad_rate_by_major":
            df.groupby("MAJOR")[C.TARGET].mean().mul(100).round(2).to_dict(),
        "grad_gpa_scientific":
            float(df[(df[C.TARGET] == 1) & (df["HS_BRANCH"] == 2)]["STD_GRADEAVG"].mean()),
        "grad_gpa_literary":
            float(df[(df[C.TARGET] == 1) & (df["HS_BRANCH"] == 1)]["STD_GRADEAVG"].mean()),
    }

    # --- Section 5.2, leakage diagnostic ------------------------------------
    out["leakage_check"] = ev.leakage_check(df)

    # --- Table 2, correlations ----------------------------------------------
    out["correlations"] = ev.correlations(
        df, C.PRE_ENROLMENT + C.IN_PROGRAMME + C.AUDIT_ONLY + C.LEAKAGE_ATTRIBUTES)

    # --- Configurations A, B, C ---------------------------------------------
    out["configurations"] = {}
    for cfg in ["A", "B", "C"]:
        plain = run_configuration(df, cfg, augment_factor=1, seed=seed)
        augmented = run_configuration(df, cfg, augment_factor=C.AUGMENT_FACTOR, seed=seed)
        out["configurations"][cfg] = {
            "no_augmentation": {k: v for k, v in plain["results"].items()},
            "with_augmentation": {k: v for k, v in augmented["results"].items()},
            "mcnemar": ev.mcnemar_tests(plain["y_test"], plain["predictions"]),
            "cross_validation": ev.cross_validate(df, C.CONFIGURATIONS[cfg], seed=seed),
            "n_train": plain["n_train"],
            "n_test": plain["n_test"],
        }

    # --- Section 5.9, fairness audit on Configuration B ----------------------
    # Recorded sex partitions the test set; it is not an input to the model.
    audited = run_configuration(df, "B", augment_factor=C.AUGMENT_FACTOR, seed=seed)
    group = df["SEX_C"].to_numpy()[audited["test_index"]]
    out["fairness"] = {
        name: ev.fairness_audit(audited["y_test"], preds, group)
        for name, preds in audited["predictions"].items()
    }

    return out
