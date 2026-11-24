"""
The experiments reported in the paper, each mapped to the table or figure it
produces.

v2.1.0 changes (response to reviewers)
  * fairness audit uses the UN-AUGMENTED Configuration B models (those in Fig. 4)
  * age imputed from the training partition / training fold only
  * McNemar p-values Holm-adjusted; CV reported without confidence intervals
  * nine-algorithm screening repeated on the training partition only
  * Section 5.11 sensitivity analyses: B-S, still-enrolled excluded, three
    outcomes, complete-follow-up cohorts, temporal validation, and a
    discrete-time survival model on admission attributes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from . import config as C
from . import evaluation as ev
from . import models
from .preprocessing import impute, missing_counts, split_scale_augment


# ------------------------------------------------------------- core runner ---
def run_features(df: pd.DataFrame, features: list[str], augment_factor: int = 1,
                 seed: int = C.SEED, target: str = C.TARGET, split=None) -> dict:
    """Fit all three models on one attribute set and one split."""
    X_tr, X_te, y_tr, y_te, test_idx = split_scale_augment(
        df, features, augment_factor=augment_factor, seed=seed,
        target=target, split=split)
    multiclass = len(np.unique(df[target])) > 2

    results, predictions, probabilities, fitted = {}, {}, {}, {}
    for model_name, model in models.build_all(seed).items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        if multiclass:
            row = ev.evaluate_multiclass(y_te, y_pred)
        else:
            y_prob = model.predict_proba(X_te)[:, 1]   # P(graduated) -> AUC
            row = ev.evaluate(y_te, y_pred, y_prob)
            probabilities[model_name] = y_prob
        row["train_accuracy"] = float(accuracy_score(y_tr, model.predict(X_tr)) * 100)
        row["importances"] = {
            f: float(round(v, 4))
            for f, v in zip(features, model.feature_importances_)
        }
        if model_name == "Decision Tree":
            row["n_nodes"] = int(model.tree_.node_count)
            row["n_leaves"] = int(model.get_n_leaves())
            row["depth"] = int(model.get_depth())

        results[model_name] = row
        predictions[model_name] = y_pred
        fitted[model_name] = model

    return {
        "features": features,
        "results": results,
        "predictions": predictions,
        "probabilities": probabilities,
        "fitted": fitted,
        "y_test": y_te,
        "test_index": test_idx,
        "n_train": len(y_tr),
        "n_test": len(y_te),
    }


def run_configuration(df: pd.DataFrame, name: str, augment_factor: int = 1,
                      seed: int = C.SEED) -> dict:
    """Backward-compatible wrapper used by figures.py."""
    return run_features(df, C.CONFIGURATIONS[name], augment_factor, seed)


def _summary(run: dict) -> dict:
    return {"n_train": run["n_train"], "n_test": run["n_test"],
            "results": run["results"]}


# ------------------------------------------------------ timing and follow-up --
def extraction_year(df: pd.DataFrame) -> int:
    if C.EXTRACTION_YEAR is not None:
        return int(C.EXTRACTION_YEAR)
    return int(df["STUDY_YEAR"].max())


def followup_table(df: pd.DataFrame) -> dict:
    """Outcome mix by admission year (Reviewer 1, point 2; Reviewer 2)."""
    ey = extraction_year(df)
    rows = {}
    first = C.FOLLOWUP_FIRST_YEAR
    year_col = df["STUDY_YEAR"].clip(lower=first).astype("Int64")
    for year, g in df.groupby(year_col):
        label = f"{first} or earlier" if int(year) == first else int(year)
        rows[label] = {
            "n": int(len(g)),
            "years_to_extraction": int(ey - year),
            "graduated_pct": float(g[C.TARGET].mean() * 100),
            "still_enrolled_pct": float(g["STILL_ENROLLED"].mean() * 100),
            "left_pct": float(((g[C.TARGET] == 0) & (g["STILL_ENROLLED"] == 0)).mean() * 100),
        }
    return {"extraction_date": C.EXTRACTION_DATE, "extraction_year_used": ey,
            "by_admission_year": rows}


def complete_followup(df: pd.DataFrame) -> pd.DataFrame:
    ey = extraction_year(df)
    keep = df["STUDY_YEAR"] <= ey - C.COMPLETE_FOLLOWUP_YEARS
    return df[keep].reset_index(drop=True)


def temporal_split(df: pd.DataFrame, share: float = C.TEMPORAL_TEST_SHARE):
    """Train on earlier admission years, test on the latest ones (~share)."""
    years = df["STUDY_YEAR"].to_numpy()
    uniq = np.sort(np.unique(years[~np.isnan(years)]))
    cutoff = uniq[-1]
    for y in uniq[::-1]:
        if (years > y).mean() >= share:
            cutoff = y
            break
    i_tr = np.where(years <= cutoff)[0]
    i_te = np.where(years > cutoff)[0]
    return (i_tr, i_te), int(cutoff)


# --------------------------------------------------------------- screening ---
def screen_algorithms(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """
    Nine-algorithm screening, five-fold CV on the TRAINING partition only.
    The hold-out partition is never touched (Reviewer 1, point 5).
    """
    from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import MinMaxScaler

    features = C.CONFIGURATIONS[C.SCREEN_CONFIGURATION]
    # Same 70/30 split as every other analysis; imputation and scaling are
    # refitted inside each CV fold by the pipeline.
    from sklearn.model_selection import train_test_split
    X = df[features].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy()
    i_tr, _ = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE,
                               random_state=seed, stratify=y)
    X, y = X[i_tr], y[i_tr]

    H = C.HYPERPARAMETERS
    algos = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(random_state=seed),
        "Neural Network": MLPClassifier(max_iter=1000, random_state=seed),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed, **H["Decision Tree"]),
        "Random Forest": RandomForestClassifier(random_state=seed, n_jobs=-1, **H["Random Forest"]),
        "GBT Classifier": GradientBoostingClassifier(random_state=seed, **H["GBT Classifier"]),
    }
    cv = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=seed)
    out = {"n_records": int(len(y)), "partition": "training only"}
    for name, est in algos.items():
        pipe = make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler(), est)
        r = cross_validate(pipe, X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        out[name] = {"accuracy_mean": float(r["test_accuracy"].mean() * 100),
                     "accuracy_sd": float(r["test_accuracy"].std(ddof=1) * 100),
                     "auc_mean": float(r["test_roc_auc"].mean())}
    return out


# ---------------------------------------------------------------- survival ---
def _period_bin(t: np.ndarray) -> np.ndarray:
    return np.clip(t, 4, 9)        # bins: <=4, 5, 6, 7, 8, 9+


def survival_model(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """
    Discrete-time, cause-specific hazard of graduation (Reviewer 3).

    Time   = regular semesters completed (at least 1).
    Event  = graduated. Still-enrolled students are right-censored at their last
             semester; students who left are censored for this cause.
    Inputs = admission attributes only (Configuration C), so nothing recorded
             after enrolment enters the model.
    Fitted on the training partition (person-period logistic regression with
    period dummies); evaluated by Harrell's C on the hold-out students.
    """
    import statsmodels.api as sm
    from sklearn.model_selection import train_test_split

    features = C.CONFIGURATIONS["C"]
    X = df[features].to_numpy(dtype=float)
    t = np.maximum(df["NORMAL_SMTR_COUNT"].to_numpy(dtype=int), 1)
    e = df[C.TARGET].to_numpy()
    i_tr, i_te = train_test_split(np.arange(len(df)), test_size=C.TEST_SIZE,
                                  random_state=seed, stratify=e)
    A, B = impute(X[i_tr], X[i_te], features)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1
    A, B = (A - mu) / sd, (B - mu) / sd

    # person-period expansion of the training students
    rows, ys, periods = [], [], []
    for a, ti, ei in zip(A, t[i_tr], e[i_tr]):
        for k in range(1, ti + 1):
            rows.append(a)
            periods.append(min(max(k, 4), 9))
            ys.append(int(ei == 1 and k == ti))
    rows, ys, periods = np.asarray(rows), np.asarray(ys), np.asarray(periods)
    levels = [4, 5, 6, 7, 8, 9]
    D = np.column_stack([(periods == lv).astype(float) for lv in levels])
    design = np.column_stack([D, rows])
    names = [f"period_{'<=4' if lv == 4 else ('9+' if lv == 9 else lv)}" for lv in levels] + features
    fit = sm.GLM(ys, design, family=sm.families.Binomial()).fit()
    beta = fit.params[len(levels):]
    ci = fit.conf_int()[len(levels):]
    risk = B @ beta
    return {
        "n_train_students": int(len(i_tr)),
        "n_person_periods": int(len(ys)),
        "events_train": int(e[i_tr].sum()),
        "odds_ratios_per_sd": {f: {"OR": float(np.exp(b)), "CI": [float(np.exp(l)), float(np.exp(h))]}
                               for f, b, (l, h) in zip(features, beta, ci)},
        "c_index_test": ev.concordance_index(t[i_te], e[i_te], risk),
        "n_test_students": int(len(i_te)),
    }


# ----------------------------------------------------------------- run all ---
def run_all(df: pd.DataFrame, seed: int = C.SEED) -> dict:
    """Every number reported in Sections 3 to 5 of the paper."""
    out = {"cohort_size": int(len(df)), "version": "2.1.0"}

    # --- Section 3.1, missing values and timing -----------------------------
    out["missing"] = missing_counts(df)
    out["followup"] = followup_table(df)

    # --- Section 3.2, descriptive -------------------------------------------
    known = df["SEX_C"] > 0
    out["descriptives"] = {
        "graduated": int(df[C.TARGET].sum()),
        "graduation_rate": float(df[C.TARGET].mean() * 100),
        "still_enrolled": int(df["STILL_ENROLLED"].sum()),
        "male_pct": float((df.loc[known, "SEX_C"] == 1).mean() * 100),
        "female_pct": float((df.loc[known, "SEX_C"] == 2).mean() * 100),
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

    # --- Section 3.4, screening on the training partition only --------------
    out["screening"] = screen_algorithms(df, seed)

    # --- Configurations A, B, B-S, C ----------------------------------------
    out["configurations"] = {}
    plain_runs = {}
    for cfg in ["A", "B", "B-S", "C"]:
        plain = run_features(df, C.CONFIGURATIONS[cfg], 1, seed)
        augmented = run_features(df, C.CONFIGURATIONS[cfg], C.AUGMENT_FACTOR, seed)
        plain_runs[cfg] = plain
        out["configurations"][cfg] = {
            "no_augmentation": plain["results"],
            "with_augmentation": augmented["results"],
            "mcnemar": ev.mcnemar_tests(plain["y_test"], plain["predictions"]),
            "cross_validation": ev.cross_validate(df, C.CONFIGURATIONS[cfg], seed=seed),
            "n_train": plain["n_train"],
            "n_test": plain["n_test"],
        }

    # --- Section 5.9, fairness audit on the UN-AUGMENTED Configuration B -----
    # Same fitted models and predictions as Fig. 4 (v2.0.0 used the augmented
    # models here, which is why its Table 7 pooled to the augmented accuracies).
    audited = plain_runs["B"]
    group = df["SEX_C"].to_numpy()[audited["test_index"]]
    out["fairness"] = {
        name: ev.fairness_audit(audited["y_test"], preds, group)
        for name, preds in audited["predictions"].items()
    }
    out["fairness_check_pooled_correct"] = {
        name: int(v["male"]["correct"] + v["female"]["correct"]
                  + int(((audited["predictions"][name] == audited["y_test"])[group == 0]).sum()))
        for name, v in out["fairness"].items()
    }

    # --- Section 5.11, sensitivity analyses ---------------------------------
    S = {}
    no_r = df[df["STILL_ENROLLED"] == 0].reset_index(drop=True)
    S["excluding_still_enrolled"] = {
        cfg: _summary(run_features(no_r, C.CONFIGURATIONS[cfg], 1, seed))
        for cfg in ["B", "B-S", "C"]}
    S["excluding_still_enrolled"]["n"] = int(len(no_r))

    S["three_outcomes"] = {
        cfg: _summary(run_features(df, C.CONFIGURATIONS[cfg], 1, seed, target="STATUS3"))
        for cfg in ["B", "C"]}

    cf = complete_followup(df)
    S["complete_followup"] = {
        "admitted_up_to": extraction_year(df) - C.COMPLETE_FOLLOWUP_YEARS,
        "n": int(len(cf)),
        "still_enrolled_in_subset": int(cf["STILL_ENROLLED"].sum()),
    }
    if len(cf) >= 50 and cf[C.TARGET].nunique() == 2:
        for cfg in ["B", "B-S", "C"]:
            S["complete_followup"][cfg] = _summary(run_features(cf, C.CONFIGURATIONS[cfg], 1, seed))

        split, cutoff = temporal_split(cf)
        S["temporal_complete_followup"] = {"train_admitted_up_to": cutoff,
                                           "test_admitted": f"{cutoff + 1}-{int(cf['STUDY_YEAR'].max())}"}
        for cfg in ["B", "B-S", "C"]:
            S["temporal_complete_followup"][cfg] = _summary(
                run_features(cf, C.CONFIGURATIONS[cfg], 1, seed, split=split))

    split, cutoff = temporal_split(df)
    S["temporal_all_cohorts"] = {"train_admitted_up_to": cutoff,
                                 "test_admitted": f"{cutoff + 1}-{int(df['STUDY_YEAR'].max())}",
                                 "note": "later test cohorts are heavily censored"}
    for cfg in ["B", "C"]:
        S["temporal_all_cohorts"][cfg] = _summary(
            run_features(df, C.CONFIGURATIONS[cfg], 1, seed, split=split))

    S["survival_admission_attributes"] = survival_model(df, seed)
    out["sensitivity"] = S
    return out
