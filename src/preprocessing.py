"""
preprocessing.py
----------------
Data preprocessing pipeline for UCAS Educational Big Data.

Covers:
  - Loading raw CSV data
  - Encoding categorical variables
  - Handling missing values
  - Normalisation
  - Train / test split (70 / 30)
  - Optional: data augmentation to simulate big data scale

Author : Samer Yaghi  |  syaghi@ucas.edu.ps  |  ORCID: 0009-0001-0268-7163
Supervisor : Prof. Rebhi S. Baraka  |  rbaraka@iugaza.edu.ps  |  Islamic University of Gaza
Institution: University College of Applied Sciences (UCAS), Gaza, Palestine
Thesis     : The Effect of Analyzing Big Educational Data on Students Performance (2022)
Thesis : The Effect of Analyzing Big Educational Data on Students' Performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ── Column constants ──────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "STUDY_TYPE_NAME",   # High school branch
    "DEPARTMENT_NAME",   # IT major
    "SEX",               # Gender
    "STD_ADDRESS",       # Student address (region)
]

NUMERIC_COLS = [
    "STD_AVG",
    "STD_GRADEAVG",
    "NORMAL_SMTR_COUNT",
    "SIFI_SMTR_COUNT",
    "PERCENTAGE",        # High school average
    "AGE",
]

TARGET_COL = "STATUS"   # 1 = Graduated, 0 = Not graduated

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# ── Encoding maps (matching thesis preprocessing) ────────────────────────────

BRANCH_MAP = {
    "literary":    1,
    "أدبي":        1,
    "human":       1,
    "scientific":  2,
    "علمي":        2,
    "other":       3,
    "أخرى":        3,
    "commercial":  3,
    "industrial":  3,
    "leadership":  3,
}

MAJOR_MAP = {
    "programming":  1,
    "برمجة":        1,
    "web":          2,
    "تصميم":        2,
    "multimedia":   3,
    "وسائط":        3,
    "gis":          4,
    "نظم":          4,
}

GENDER_MAP = {
    "m": 1, "male": 1, "ذكر": 1,
    "f": 2, "female": 2, "أنثى": 2,
}

ADDRESS_MAP = {
    "north gaza":   1, "شمال غزة": 1,
    "gaza":         2, "غزة":      2,
    "middle":       3, "وسط":      3,
    "khan yunis":   4, "خان يونس": 4,
    "rafah":        5, "رفح":      5,
}

STATUS_MAP = {
    "g": 1, "خريج": 1, "graduated": 1,
    "r": 0, "منتظم":  0,
    "d": 0, "مفصول":  0,
    "c": 0, "منقطع":  0,
    "w": 0, "منسحب":  0,
    "s": 0, "مؤجل":   0,
    "m": 0, "متوفى":  0,
}


# ── Main preprocessing class ──────────────────────────────────────────────────

class UCASPreprocessor:
    """Full preprocessing pipeline for UCAS student dataset."""

    def __init__(self, augment_factor: int = 4):
        """
        Parameters
        ----------
        augment_factor : int
            Multiply dataset size to simulate big data (thesis used 4x:
            original 4,704 records → 18,816 records).
        """
        self.augment_factor = augment_factor
        self.scaler = MinMaxScaler()
        self.label_encoders: dict = {}
        self.feature_cols_used: list = []

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def load(self, csv_path: str) -> pd.DataFrame:
        """Load raw CSV and apply basic type coercions."""
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df.columns = [c.strip().upper() for c in df.columns]
        print(f"[load] Loaded {len(df):,} records, {len(df.columns)} columns.")
        return df

    # ── Step 2: Clean ─────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove students who enrolled but never attended any course.
        (Matches thesis: 'All students who applied but did not study were deleted.')
        """
        before = len(df)
        # Drop if both course counts are zero / null
        df = df[~((df.get("NORMAL_SMTR_COUNT", pd.Series([1])) == 0) &
                  (df.get("SIFI_SMTR_COUNT",  pd.Series([1])) == 0))]
        df = df.dropna(subset=[TARGET_COL])
        after = len(df)
        print(f"[clean] Removed {before - after:,} invalid records → {after:,} remain.")
        return df.reset_index(drop=True)

    # ── Step 3: Encode ────────────────────────────────────────────────────────

    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map categorical text values to numeric codes."""
        df = df.copy()

        # High school branch
        if "STUDY_TYPE_NAME" in df.columns:
            df["STUDY_TYPE_NAME"] = (
                df["STUDY_TYPE_NAME"].str.lower().str.strip()
                .map(lambda x: next((v for k, v in BRANCH_MAP.items() if k in x), 3))
            )

        # IT major
        if "DEPARTMENT_NAME" in df.columns:
            df["DEPARTMENT_NAME"] = (
                df["DEPARTMENT_NAME"].str.lower().str.strip()
                .map(lambda x: next((v for k, v in MAJOR_MAP.items() if k in x), 1))
            )

        # Gender
        if "SEX" in df.columns:
            df["SEX"] = (
                df["SEX"].str.lower().str.strip()
                .map(GENDER_MAP).fillna(1).astype(int)
            )

        # Address / region
        if "STD_ADDRESS" in df.columns:
            df["STD_ADDRESS"] = (
                df["STD_ADDRESS"].str.lower().str.strip()
                .map(lambda x: next((v for k, v in ADDRESS_MAP.items() if k in x), 2))
            )

        # Target label
        df[TARGET_COL] = (
            df[TARGET_COL].astype(str).str.lower().str.strip()
            .map(STATUS_MAP).fillna(0).astype(int)
        )

        print(f"[encode] Graduation rate: "
              f"{df[TARGET_COL].mean()*100:.1f}% "
              f"({df[TARGET_COL].sum():,} / {len(df):,})")
        return df

    # ── Step 4: Normalise ─────────────────────────────────────────────────────

    def normalise(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Min-max normalise numeric features."""
        df = df.copy()
        num_cols = [c for c in NUMERIC_COLS if c in df.columns]
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        if fit:
            df[num_cols] = self.scaler.fit_transform(df[num_cols])
        else:
            df[num_cols] = self.scaler.transform(df[num_cols])
        return df

    # ── Step 5: Augment ───────────────────────────────────────────────────────

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Multiply the dataset to simulate big data scale.
        Thesis: original 4,704 records were amplified to 18,896
        (factor ≈ 4) with minor Gaussian noise on numeric columns.
        """
        if self.augment_factor <= 1:
            return df
        num_cols = [c for c in NUMERIC_COLS if c in df.columns]
        copies = [df]
        rng = np.random.default_rng(seed=42)
        for _ in range(self.augment_factor - 1):
            noisy = df.copy()
            noise = rng.normal(0, 0.01, size=(len(df), len(num_cols)))
            noisy[num_cols] = np.clip(noisy[num_cols].values + noise, 0, 1)
            copies.append(noisy)
        big = pd.concat(copies, ignore_index=True)
        print(f"[augment] Dataset amplified: {len(df):,} → {len(big):,} records.")
        return big

    # ── Step 6: Split ─────────────────────────────────────────────────────────

    def split(self, df: pd.DataFrame):
        """70 / 30 stratified train-test split."""
        available = [c for c in FEATURE_COLS if c in df.columns]
        self.feature_cols_used = available
        X = df[available].values
        y = df[TARGET_COL].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        print(f"[split] Train: {len(X_train):,}  |  Test: {len(X_test):,}")
        return X_train, X_test, y_train, y_test

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def run(self, csv_path: str):
        """Execute all preprocessing steps and return train/test arrays."""
        df = self.load(csv_path)
        df = self.clean(df)
        df = self.encode(df)
        df = self.normalise(df, fit=True)
        df = self.augment(df)
        return self.split(df)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="UCAS data preprocessing")
    parser.add_argument("--data", default="data/sample_data.csv")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    prep = UCASPreprocessor(augment_factor=1 if args.no_augment else 4)
    X_train, X_test, y_train, y_test = prep.run(args.data)
    print(f"\nFeatures used ({len(prep.feature_cols_used)}): {prep.feature_cols_used}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
