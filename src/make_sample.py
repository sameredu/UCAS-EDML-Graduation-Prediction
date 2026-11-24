"""
Generate the synthetic sample shipped with this repository.

WHY SYNTHETIC. The analytical cohort is real student data. Even with names and
identifiers removed, the remaining columns re-identify: on the full
quasi-identifier set (recorded sex, age, specialisation, enrolment year, region,
high-school branch, certificate year) roughly three records in four are unique,
and on recorded sex plus specialisation plus the exact cumulative GPA nearly
nine in ten are. Anyone who knows a former classmate's enrolment year, major and
approximate standing could recover that person's exact GPA and whether they
graduated. A publishable sample therefore cannot be drawn from the real records,
however it is anonymised.

WHAT THIS PRODUCES. Records that correspond to no real student, drawn from
marginal distributions estimated on the cohort, and carrying the structural
properties the repository exists to demonstrate:

  * the class balance of the real cohort
  * GRADUATE_YEAR populated if and only if the student graduated
  * STD_GRADEAVG equal to STD_AVG for every non-graduate and for about 55% of
    graduates; a differing value occurs only among graduates (v2.1.0; the
    v2.0.0 sample gave every graduate a differing value, unlike the real data,
    where 1,002 of 1,826 graduates match)
  * a positive association between semesters completed and graduation
  * still-enrolled students concentrated in the most recent admission years
  * a few missing ages and unrecorded sex values, so the imputation and
    exclusion rules are exercised

Point estimates from the synthetic sample will NOT match the paper. The sample
exists to exercise the code and to let a reader confirm the leakage mechanism,
not to reproduce the results; for those, use the real extract.

    python -m src.make_sample --out data/sample_data.csv --n 400
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C

DEPARTMENTS = ["البرمجيات وقواعد البيانات", "تصميم وتطوير مواقع الويب",
               "تكنولوجيا الوسائط المتعددة", "نظم المعلومات الجغرافية",
               "البرمجة وتحليل النظم"]
DEPT_P = [0.17, 0.15, 0.53, 0.10, 0.05]

BRANCHES = ["علوم انسانية", "علمي", "الفرع الشرعي", "أدبي - قديم", "صناعي"]
BRANCH_P = [0.66, 0.28, 0.02, 0.02, 0.02]

REGIONS = ["غزة", "شمال غزة", "المنطقة الوسطى", "خان يونس", "رفح"]
REGION_P = [0.46, 0.22, 0.13, 0.13, 0.06]

# Outcome mix of the real cohort (Section 3.2 of the paper).
STATUSES = ["G", "R", "C", "D", "S"]
STATUS_P = [0.3882, 0.2183, 0.2173, 0.1569, 0.0193]  # sums to 1.0


def generate(n: int = 400, seed: int = C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    study_year = rng.integers(2000, 2020, size=n)
    # Still-enrolled students belong mostly to recent cohorts, as in any
    # extract taken at a single date.
    status = rng.choice(STATUSES, size=n, p=STATUS_P)
    recent = study_year >= 2015
    old_r = (status == "R") & ~recent
    status[old_r] = rng.choice(["G", "C", "D"], size=old_r.sum(), p=[0.5, 0.3, 0.2])
    n_r_target = int(round(STATUS_P[1] * n))
    pool = np.where(recent & (status != "R"))[0]
    need = max(0, n_r_target - int((status == "R").sum()))
    status[rng.choice(pool, size=min(need, len(pool)), replace=False)] = "R"
    graduated = status == "G"

    sex = rng.choice(["M", "F"], size=n, p=[0.735, 0.265])
    branch = rng.choice(BRANCHES, size=n, p=BRANCH_P)
    dept = rng.choice(DEPARTMENTS, size=n, p=DEPT_P)
    region = rng.choice(REGIONS, size=n, p=REGION_P)

    taw_year = study_year - rng.integers(0, 3, size=n)
    age = np.clip(rng.normal(21.5, 3.0, size=n), 18, 45).round(0)

    hs_score = np.clip(rng.normal(74, 10, size=n), 50, 99).round(1)

    # Graduates complete more semesters and carry higher averages. The gap is
    # what makes the corrected problem learnable at all.
    regular = np.where(graduated,
                       rng.integers(5, 13, size=n),
                       rng.integers(1, 7, size=n))
    summer = np.where(graduated,
                      rng.integers(0, 4, size=n),
                      rng.integers(0, 3, size=n))
    std_avg = np.where(graduated,
                       rng.normal(76, 7, size=n),
                       rng.normal(64, 9, size=n))
    std_avg = np.clip(std_avg, 35, 99).round(2)

    # ---- the two leakage attributes, reproduced exactly as in the source system
    # STD_GRADEAVG differs from STD_AVG only once a student has graduated.
    # About 55% of graduates keep a graduate average equal to the cumulative one.
    differs = graduated & (rng.random(n) > 0.55)
    grade_avg = np.where(differs,
                         np.clip(std_avg + rng.uniform(0.1, 3, size=n), 35, 100).round(2),
                         std_avg)
    # GRADUATE_YEAR is recorded only for graduates.
    grad_year = np.where(graduated, study_year + rng.integers(3, 7, size=n), np.nan)

    age = age.astype(float)
    age[rng.random(n) < 0.02] = np.nan            # a few missing ages
    sex = sex.astype(object)
    sex[rng.random(n) < 0.01] = None               # a few unrecorded values

    return pd.DataFrame({
        "SEX": sex,
        "AGE": age,
        "STUDY_TYPE_NAME": branch,
        "STUDY_YEAR": study_year.astype(float),
        "STD_AVG": std_avg,
        "DEPARTMENT_NAME": dept,
        "STD_GRADEAVG": grade_avg,
        "STATUS": status,
        "STD_ADDRESS": region,
        "PERCENTAGE": hs_score,
        "TAW_YEAR": taw_year.astype(float),
        "NORMAL_SMTR_COUNT": regular,
        "SIFI_SMTR_COUNT": summer,
        "GRADUATE_YEAR": grad_year,
    })


def verify(df: pd.DataFrame) -> None:
    """Check that the synthetic sample carries the structure it is meant to."""
    grad = df["STATUS"] == "G"
    assert df.loc[grad, "GRADUATE_YEAR"].notna().all(), \
        "every graduate must have a graduation year"
    assert df.loc[~grad, "GRADUATE_YEAR"].isna().all(), \
        "no non-graduate may have a graduation year"
    assert np.isclose(df.loc[~grad, "STD_GRADEAVG"], df.loc[~grad, "STD_AVG"]).all(), \
        "graduate GPA must equal cumulative GPA for non-graduates"
    same_g = np.isclose(df.loc[grad, "STD_GRADEAVG"], df.loc[grad, "STD_AVG"])
    print(f"  structure verified: GRADUATE_YEAR iff graduated; "
          f"graduates with matching graduate GPA: {same_g.sum()} of {grad.sum()}")
    print(f"  graduation rate: {grad.mean() * 100:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample_data.csv")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()

    df = generate(args.n, args.seed)
    verify(df)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"  {len(df)} synthetic records written to {args.out}")


if __name__ == "__main__":
    main()
