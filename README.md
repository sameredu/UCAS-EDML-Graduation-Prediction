# UCAS Educational Big Data — ML Graduation Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/Apache%20Spark-3.x-orange)](https://spark.apache.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Companion code for:**
> Yaghi, S. A., & Baraka, R. S. (2022). *Predicting Students' Academic Performance Using Big Educational Data and Machine Learning: A Case Study of UCAS*. Master's Thesis, The Islamic University of Gaza.

---

## Overview

This repository contains the full PySpark implementation of the machine learning framework presented in the thesis. It applies three classification algorithms — **Decision Tree (DT)**, **Random Forest (RF)**, and **Gradient Boosting Tree (GBT)** — to predict graduation outcomes for IT-major students at the University College of Applied Sciences (UCAS), Gaza, Palestine.

The dataset covers **19 years of student records (2000–2019)** across four IT specialisations: Multimedia, Programming & Databases, Web Design & Development, and GIS.

---

## Key Results

| Algorithm | Accuracy (CA) | AUC   | F1    | Precision | Recall |
|-----------|--------------|-------|-------|-----------|--------|
| **Random Forest** | **99.8%** | **0.999** | **0.998** | **0.998** | **0.998** |
| Decision Tree | 97.1% | 0.979 | 0.971 | 0.972 | 0.971 |
| GBT Classifier | 93.0% | 0.965 | 0.923 | 0.931 | 0.930 |

**Top graduation predictors (Pearson correlation):**
- Student cumulative GPA (`STD_AVG`): r = +0.997
- Number of regular semesters (`NORMAL_SMTR_COUNT`): r = +0.489
- Number of summer semesters (`SIFI_SMTR_COUNT`): r = +0.266

**Spark acceleration:**
- Random Forest: 75% faster (60s → 15s)
- GBT Classifier: 72.9% faster
- Decision Tree: 58.9% faster

---

## Repository Structure

```
ucas-edml/
├── data/
│   ├── sample_data.csv          # Anonymised sample dataset (100 records)
│   └── data_schema.md           # Full schema description (21 attributes)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb   # EDA — 19 years of registration data
│   ├── 02_preprocessing.ipynb          # Cleaning, encoding, normalisation
│   ├── 03_ml_models.ipynb              # DT, RF, GBT comparison
│   └── 04_spark_acceleration.ipynb     # Spark cluster execution
├── src/
│   ├── preprocessing.py         # Data preprocessing pipeline
│   ├── decision_tree.py         # DT model (standalone Python + PySpark)
│   ├── random_forest.py         # RF model (standalone Python + PySpark)
│   ├── gbt_classifier.py        # GBT model (standalone Python + PySpark)
│   ├── spark_pipeline.py        # Full Spark ML pipeline
│   └── evaluation.py            # Confusion matrix, metrics, visualisation
├── results/
│   └── confusion_matrices/      # Saved confusion matrix plots
├── docs/
│   └── spark_setup.md           # Spark cluster setup guide
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A — Without Spark (scikit-learn, recommended for testing)

```bash
pip install -r requirements.txt
python src/random_forest.py --data data/sample_data.csv
```

### Option B — With Apache Spark (PySpark)

```bash
# Start Spark master
spark-class org.apache.spark.deploy.master.Master

# Start worker(s)
spark-class org.apache.spark.deploy.worker.Worker spark://MASTER_IP:7077

# Submit job
spark-submit src/spark_pipeline.py --data data/sample_data.csv
```

### Run Jupyter notebooks

```bash
jupyter notebook notebooks/
```

---

## Dataset Schema

The full dataset contains **21 attributes** per student record. Due to privacy restrictions, only an anonymised 100-record sample is included. See `data/data_schema.md` for the complete schema.

| # | Attribute | Type | Description |
|---|-----------|------|-------------|
| 1 | `STUDY_TYPE_NAME` | Categorical | High school branch (1=Literary, 2=Scientific, 3=Other) |
| 2 | `DEPARTMENT_NAME` | Categorical | IT major (1=Prog&DB, 2=Web, 3=Multimedia, 4=GIS) |
| 3 | `STD_AVG` | Float | University cumulative GPA [35–100] |
| 4 | `STD_GRADEAVG` | Float | Graduate average [60–100] |
| 5 | `NORMAL_SMTR_COUNT` | Int | Number of regular semesters [0–12] |
| 6 | `SIFI_SMTR_COUNT` | Int | Number of summer semesters [0–4] |
| 7 | `SEX` | Binary | Gender (1=Male, 2=Female) |
| 8 | `PERCENTAGE` | Float | High school average [50–100] |
| 9 | `AGE` | Int | Student age [18–60] |
| ... | ... | ... | ... |
| 17 | **`STATUS`** | **Binary** | **Target: Graduated=1, Not graduated=0** |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{yaghi2022ucas,
  author    = {Yaghi, Samer A.},
  title     = {The Effect of Analyzing Big Educational Data on Students' Performance:
               The Case of University College of Applied Sciences},
  school    = {The Islamic University of Gaza},
  year      = {2022},
  type      = {Master's Thesis},
  address   = {Gaza, Palestine}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

**Samer A. Yaghi** · The Islamic University of Gaza · Palestine
