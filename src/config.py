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

# EXCLUDED ON LEAKAGE GROUNDS. Both are filled in as a consequence of graduating:
#   GRADUATE_YEAR  is recorded for every graduate and for no one else, so it
#                  determines the outcome exactly
#   STD_GRADEAVG   equals STD_AVG for every non-graduate and for some graduates;
#                  a DIFFERING value occurs only among graduates, so the field is
#                  partly determined by the outcome
# They are modelled only in Configuration A, which exists to demonstrate the
# artefact and is never presented as a result.
LEAKAGE_ATTRIBUTES = ["STD_GRADEAVG", "GRAD_YEAR_F"]

# Recorded sex is held OUT of every attribute set. It is a binary administrative
# field, it is not self-reported, and no prediction is conditioned on it. It is
# retained solely to partition the test set for the fairness audit.
AUDIT_ONLY = ["SEX_C"]

# SECOND, PARTIAL FORM OF LEAKAGE (v2.1.0). The semester counts are known at any
# point during enrolment, but in this extract they are counted up to the
# extraction date, so a graduate's count reflects a completed programme.
# Configuration B therefore classifies STATUS AT EXTRACTION; it is not a
# forecast. Configuration B-S removes the counts.
SEMESTER_COUNTS = ["NORMAL_SMTR_COUNT", "SIFI_SMTR_COUNT"]

CONFIGURATIONS = {
    "A": PRE_ENROLMENT + IN_PROGRAMME + LEAKAGE_ATTRIBUTES,  # leakage diagnostic
    "B": PRE_ENROLMENT + IN_PROGRAMME,                       # status at extraction (main)
    "B-S": PRE_ENROLMENT + ["STD_AVG"],                      # B without semester counts
    "C": PRE_ENROLMENT,                                      # admission-time only
}

CONFIG_LABELS = {
    "A": "A: with leakage attributes",
    "B": "B: corrected, status at extraction",
    "B-S": "B-S: B without semester counts",
    "C": "C: pre-enrolment only",
}

# Attributes with missing values that are imputed INSIDE the pipeline, from the
# training partition (or training fold) only. v2.0.0 imputed age with the
# median of the whole cohort before the split.
IMPUTE_MEDIAN = ["AGE"]

# Tolerance used when comparing STD_GRADEAVG with STD_AVG (numpy.isclose).
GPA_RTOL = 1e-05
GPA_ATOL = 1e-08

# ------------------------------------------------------ outcome and timing ---
STILL_ENROLLED_CODE = "R"          # STATUS code: still enrolled at extraction
# Extraction date of the institutional extract; used only for the follow-up
# table and the complete-follow-up cutoff.
EXTRACTION_DATE = "mid-2020"       # latest recorded graduation year is 2020
EXTRACTION_YEAR = 2020
# Cohorts admitted at least this many years before extraction are treated as
# having complete follow-up in the sensitivity analysis.
COMPLETE_FOLLOWUP_YEARS = 6     # i.e. admitted up to 2014
# Temporal validation: the latest admission years are held out. TEMPORAL_TEST_SHARE
# is the share of the complete-follow-up cohort placed in the test set.
TEMPORAL_TEST_SHARE = 0.30
# Admission years before this are grouped into one row of the follow-up table;
# the field holds some data-entry errors and is not an input to any model.
FOLLOWUP_FIRST_YEAR = 2000

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

# ------------------------------------------------- nine-algorithm screening ---
# v2.0.0 carried the choice of three algorithms over from the earlier analysis.
# v2.1.0 repeats the screening with five-fold CV on the TRAINING partition only;
# the hold-out set is not used for model selection.
SCREEN_CONFIGURATION = "B"

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
