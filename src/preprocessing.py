"""
Cohort construction and preprocessing.

ORDERING NOTE — this is the correction at the heart of the paper.

The released version of the earlier implementation normalised and augmented the
data BEFORE splitting it, so near-duplicate copies of training records appeared
in the test partition. Here the order is:

    1. build the cohort
    2. split  (stratified 70/30)
    3. fit the scaler on the TRAINING partition only
    4. augment the TRAINING partition only

`split_scale_augment()` is the only function that produces model-ready arrays,
and it enforces that order. There is no code path that augments before splitting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import config as C


# --------------------------------------------------------------- loading ----
def load_raw(path: str) -> pd.DataFrame:
    """Read the extract. Accepts .csv, .xls/.xlsx or a pandas pickle."""
    if path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def _branch(value) -> int:
    v = str(value).strip()
    if v in C.HS_SCIENTIFIC:
        return 2
    if v in C.HS_LITERARY:
        return 1
    return 3


def build_cohort(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Three documented filters, applied in order. Each step prints its count so the
    cohort reported in the paper can be checked line by line.
    """
    n0 = len(df)

    step1 = df[df["DEPARTMENT_NAME"].isin(C.IT_DEPARTMENTS)].copy()
    # Applicants who never commenced study: no regular and no summer semesters.
    step2 = step1[~((step1["NORMAL_SMTR_COUNT"] == 0) & (step1["SIFI_SMTR_COUNT"] == 0))]
    step3 = step2[step2["STD_AVG"].notna()].reset_index(drop=True)

    if verbose:
        print(f"  full extract                              {n0:>7,}")
        print(f"  IT specialisations                        {len(step1):>7,}")
        print(f"  with at least one semester                {len(step2):>7,}")
        print(f"  with a recorded cumulative average        {len(step3):>7,}")

    return step3


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Map the source columns onto the modelling attributes."""
    out = df.copy()
    out["MAJOR"] = out["DEPARTMENT_NAME"].map(C.IT_DEPARTMENTS)
    out["HS_BRANCH"] = out["STUDY_TYPE_NAME"].map(_branch)
    out["REGION"] = out["STD_ADDRESS"].map(C.REGIONS).fillna(C.REGION_UNKNOWN).astype(int)
    out["AGE"] = out["AGE"].fillna(out["AGE"].median())
    out = out.rename(columns={"PERCENTAGE": "HS_SCORE"})

    # Audit-only. Never enters an attribute set; see config.AUDIT_ONLY.
    out["SEX_C"] = out["SEX"].map({"M": 1, "F": 2}).fillna(1).astype(int)

    # Leakage attributes, materialised only so that Configuration A can
    # demonstrate the artefact. GRAD_YEAR_F is 0 exactly when the student did
    # not graduate, which is why it reproduces the label perfectly.
    out["GRAD_YEAR_F"] = out["GRADUATE_YEAR"].fillna(0)

    out[C.TARGET] = (out["STATUS"] == C.GRADUATED_CODE).astype(int)
    return out


def prepare(path: str, verbose: bool = True) -> pd.DataFrame:
    """load → cohort → encode. Returns one row per student."""
    if verbose:
        print("Cohort construction")
    df = build_cohort(load_raw(path), verbose=verbose)
    return encode(df)


# ------------------------------------------------------------ augmentation ---
def augment(X: np.ndarray, y: np.ndarray, numeric_idx: list[int],
            factor: int = C.AUGMENT_FACTOR, sigma: float = C.AUGMENT_SIGMA,
            seed: int = C.SEED):
    """
    Replicate the rows `factor` times, adding Gaussian noise to the scaled
    numeric columns. Applied to the TRAINING partition only.

    Reported in the paper as a negative control: it reduces accuracy for all
    three classifiers, so the un-augmented results are the primary ones.
    """
    if factor <= 1:
        return X, y
    rng = np.random.default_rng(seed)
    Xs, ys = [X], [y]
    for _ in range(factor - 1):
        Xn = X.copy()
        noise = rng.normal(0, sigma, size=(X.shape[0], len(numeric_idx)))
        Xn[:, numeric_idx] = np.clip(Xn[:, numeric_idx] + noise, 0, 1)
        Xs.append(Xn)
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)


# ------------------------------------------------------------- the pipeline --
def split_scale_augment(df: pd.DataFrame, features: list[str],
                        augment_factor: int = 1, seed: int = C.SEED):
    """
    The leakage-free pipeline, in the only order this module permits.

    Returns (X_train, X_test, y_train, y_test, test_index).
    `test_index` indexes back into `df` so the fairness audit can partition the
    test set by an attribute the model never saw.
    """
    for f in features:
        if f in C.AUDIT_ONLY:
            raise ValueError(
                f"{f} is audit-only and must not be used as a model input"
            )

    X = df[features].to_numpy(dtype=float)
    y = df[C.TARGET].to_numpy()
    idx = np.arange(len(df))

    # 1. SPLIT FIRST — before any transformation touches the data.
    i_tr, i_te = train_test_split(
        idx, test_size=C.TEST_SIZE, random_state=seed, stratify=y
    )

    # 2. Fit the scaler on the training partition only.
    scaler = MinMaxScaler().fit(X[i_tr])
    X_tr, X_te = scaler.transform(X[i_tr]), scaler.transform(X[i_te])
    y_tr, y_te = y[i_tr], y[i_te]

    # 3. Augment the training partition only.
    numeric_idx = [i for i, f in enumerate(features) if f in C.NUMERIC]
    X_tr, y_tr = augment(X_tr, y_tr, numeric_idx, factor=augment_factor, seed=seed)

    return X_tr, X_te, y_tr, y_te, i_te
