"""
Cohort construction and preprocessing.

ORDERING NOTE — this is the correction at the heart of the paper.

The released version of the earlier implementation normalised and augmented the
data BEFORE splitting it, so near-duplicate copies of training records appeared
in the test partition. Here the order is:

    1. build the cohort
    2. split  (stratified 70/30)
    3. impute missing age with the TRAINING-partition median   (v2.1.0)
    4. fit the scaler on the TRAINING partition only
    5. augment the TRAINING partition only

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
    # Age is NOT imputed here. v2.0.0 filled it with the median of the whole
    # cohort before the split; v2.1.0 imputes inside the pipeline from the
    # training partition (or training fold) only. See split_scale_augment().
    out = out.rename(columns={"PERCENTAGE": "HS_SCORE"})

    # Audit-only. Never enters an attribute set; see config.AUDIT_ONLY.
    # Missing or unrecognised values are coded 0 (unknown) and excluded from
    # the subgroup audit. v2.0.0 assigned them to the male category.
    out["SEX_C"] = (out["SEX"].astype(str).str.strip().str.upper()
                    .map({"M": 1, "F": 2}).fillna(0).astype(int))

    # Leakage attributes, materialised only so that Configuration A can
    # demonstrate the artefact. GRAD_YEAR_F is 0 exactly when the student did
    # not graduate, which is why it reproduces the label perfectly.
    out["GRAD_YEAR_F"] = out["GRADUATE_YEAR"].fillna(0)

    # Keep the raw status code: C.TARGET is also called "STATUS" and is
    # overwritten with the binary outcome on the next line.
    out["STATUS_CODE"] = out["STATUS"].astype(str).str.strip()
    out["STILL_ENROLLED"] = (out["STATUS_CODE"] == C.STILL_ENROLLED_CODE).astype(int)
    out[C.TARGET] = (out["STATUS_CODE"] == C.GRADUATED_CODE).astype(int)
    # Three outcomes at extraction: 0 left without graduating, 1 graduated,
    # 2 still enrolled (right-censored).
    out["STATUS3"] = np.where(out[C.TARGET] == 1, 1,
                              np.where(out["STILL_ENROLLED"] == 1, 2, 0))
    return out


def missing_counts(df: pd.DataFrame) -> dict:
    """How many records each missing-value rule touches (Reviewer 1, point 5)."""
    return {
        "age_missing": int(df["AGE"].isna().sum()),
        "sex_unknown": int((df["SEX_C"] == 0).sum()),
        "region_unknown": int((df["REGION"] == C.REGION_UNKNOWN).sum()),
        "hs_score_missing": int(df["HS_SCORE"].isna().sum()),
    }


def impute(X_tr: np.ndarray, X_te: np.ndarray, features: list[str]):
    """Median imputation fitted on the training rows only."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for j, f in enumerate(features):
        if f in C.IMPUTE_MEDIAN or np.isnan(X_tr[:, j]).any() or np.isnan(X_te[:, j]).any():
            med = np.nanmedian(X_tr[:, j])
            X_tr[np.isnan(X_tr[:, j]), j] = med
            X_te[np.isnan(X_te[:, j]), j] = med
    return X_tr, X_te


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
                        augment_factor: int = 1, seed: int = C.SEED,
                        target: str = C.TARGET, split=None):
    """
    The leakage-free pipeline, in the only order this module permits.

    Returns (X_train, X_test, y_train, y_test, test_index).
    `test_index` indexes back into `df` so the fairness audit can partition the
    test set by an attribute the model never saw.

    `split`, if given, is a (train_idx, test_idx) pair used instead of the
    stratified random split (temporal validation).
    """
    for f in features:
        if f in C.AUDIT_ONLY:
            raise ValueError(
                f"{f} is audit-only and must not be used as a model input"
            )

    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy()
    idx = np.arange(len(df))

    # 1. SPLIT FIRST — before any transformation touches the data.
    if split is None:
        i_tr, i_te = train_test_split(
            idx, test_size=C.TEST_SIZE, random_state=seed, stratify=y
        )
    else:
        i_tr, i_te = split

    # 2. Impute from the training partition only.
    X_tr, X_te = impute(X[i_tr], X[i_te], features)

    # 3. Fit the scaler on the training partition only.
    scaler = MinMaxScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    y_tr, y_te = y[i_tr], y[i_te]

    # 4. Augment the training partition only.
    numeric_idx = [i for i, f in enumerate(features) if f in C.NUMERIC]
    X_tr, y_tr = augment(X_tr, y_tr, numeric_idx, factor=augment_factor, seed=seed)

    return X_tr, X_te, y_tr, y_te, i_te
