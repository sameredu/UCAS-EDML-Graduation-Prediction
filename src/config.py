"""
Configuration for the UCAS graduation-prediction study.

Everything that defines an experiment lives here: the cohort filter, the three
attribute configurations, the hyperparameters and the random seed. Nothing in
this file is derived from the outcome.
"""

SEED = 42
TEST_SIZE = 0.30
CV_FOLDS = 5
AUGMENT_FACTOR = 4
AUGMENT_SIGMA = 0.01

# ---------------------------------------------------------------- cohort ----
# The four IT specialisations. "Programming and Systems Analysis" is the
# predecessor programme of "Software and Databases" and is mapped to the same code.
IT_DEPARTMENTS = {
    "البرمجيات وقواعد البيانات": 1,   # Software and Databases
    "البرمجة وتحليل النظم": 1,        # Programming and Systems Analysis
    "تصميم وتطوير مواقع الويب": 2,    # Web Design and Development
    "تكنولوجيا الوسائط المتعددة": 3,  # Multimedia Technology
    "نظم المعلومات الجغرافية": 4,     # Geographic Information Systems
}

HS_SCIENTIFIC = ["علمي", "علمي - قديم", "علمي أزهري", "العلوم الطبيعية", "علوم"]
HS_LITERARY = ["علوم انسانية", "أدبي - قديم", "أدبي أزهري", "الفرع الشرعي", "أدبي"]

REGIONS = {
    "شمال غزة": 1,       # North Gaza
    "غزة": 2,            # Gaza
    "المنطقة الوسطى": 3,  # Middle Area
    "خان يونس": 4,       # Khan Yunis
    "رفح": 5,            # Rafah
}
REGION_UNKNOWN = 6

GRADUATED_CODE = "G"

# ------------------------------------------------------------- attributes ----
# Available before a student enrols.
PRE_ENROLMENT = ["AGE", "HS_BRANCH", "MAJOR", "REGION", "HS_SCORE"]

# Accumulate during study.
IN_PROGRAMME = ["NORMAL_SMTR_COUNT", "SIFI_SMTR_COUNT", "STD_AVG"]

# EXCLUDED ON LEAKAGE GROUNDS. Both are deterministic functions of the outcome:
#   GRADUATE_YEAR  is NULL for every student who did not graduate
#   STD_GRADEAVG   equals STD_AVG for every student who did not graduate
# They are modelled only in Configuration A, which exists to demonstrate the
# artefact and is never presented as a result.
LEAKAGE_ATTRIBUTES = ["STD_GRADEAVG", "GRAD_YEAR_F"]

# Recorded sex is held OUT of every attribute set. It is a binary administrative
# field, it is not self-reported, and no prediction is conditioned on it. It is
# retained solely to partition the test set for the fairness audit.
AUDIT_ONLY = ["SEX_C"]

CONFIGURATIONS = {
    "A": PRE_ENROLMENT + IN_PROGRAMME + LEAKAGE_ATTRIBUTES,  # leakage diagnostic
    "B": PRE_ENROLMENT + IN_PROGRAMME,                       # corrected (main)
    "C": PRE_ENROLMENT,                                      # admission-time only
}

CONFIG_LABELS = {
    "A": "A: with leakage attributes",
    "B": "B: corrected (full record)",
    "C": "C: pre-enrolment only",
}

# Columns that are min-max scaled. Categorical codes are left as integers,
# which tree-based models treat identically either way.
NUMERIC = ["AGE", "HS_SCORE", "NORMAL_SMTR_COUNT", "SIFI_SMTR_COUNT",
           "STD_AVG", "STD_GRADEAVG", "GRAD_YEAR_F"]

TARGET = "STATUS"

DISPLAY_NAMES = {
    "AGE": "Age at enrolment",
    "HS_BRANCH": "High-school branch",
    "MAJOR": "IT major",
    "REGION": "Region of residence",
    "HS_SCORE": "High-school score",
    "NORMAL_SMTR_COUNT": "Regular semesters",
    "SIFI_SMTR_COUNT": "Summer semesters",
    "STD_AVG": "Cumulative GPA",
    "STD_GRADEAVG": "Graduate GPA (leakage)",
    "GRAD_YEAR_F": "Graduation year (leakage)",
    "SEX_C": "Recorded sex",
}

# ---------------------------------------------------------- hyperparameters ---
# Held fixed at the values used in the earlier analysis, so that the change in
# performance between configurations is attributable to the attribute set
# rather than to tuning.
HYPERPARAMETERS = {
    "Random Forest": dict(n_estimators=100, max_depth=None, max_features="sqrt",
                          min_samples_split=4, min_samples_leaf=2,
                          criterion="gini", bootstrap=True),
    "Decision Tree": dict(max_depth=6, min_samples_split=4, min_samples_leaf=2,
                          criterion="gini"),
    "GBT Classifier": dict(n_estimators=100, max_depth=3, learning_rate=0.1,
                           subsample=0.8, min_samples_split=4, min_samples_leaf=2),
}

MODEL_ORDER = ["Random Forest", "Decision Tree", "GBT Classifier"]
