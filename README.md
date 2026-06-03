# UCAS Educational Big Data — ML Graduation Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://python.org)
[![PySpark](https://img.shields.io/badge/Apache%20Spark-3.x-orange)](https://spark.apache.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--0268--7163-brightgreen)](https://orcid.org/0009-0001-0268-7163)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-Samer--Yaghi-00CCBB)](https://www.researchgate.net/profile/Samer-Yaghi-2)

> **Companion code for:**
>
> Yaghi, S., & Baraka, R. (2022). *The Effect of Analyzing Big Educational Data on Students' Performance: The Case of University College of Applied Sciences*. Master's Thesis, Islamic University of Gaza.

---

## Authors

| Name | Affiliation | Email | ORCID |
|------|------------|-------|-------|
| **Samer Yaghi** | University College of Applied Sciences (UCAS) · Islamic University of Gaza | syaghi@ucas.edu.ps | [0009-0001-0268-7163](https://orcid.org/0009-0001-0268-7163) |
| **Prof. Rebhi S. Baraka** | Islamic University of Gaza | rbaraka@iugaza.edu.ps | — |

🔗 ResearchGate: [Samer Yaghi](https://www.researchgate.net/profile/Samer-Yaghi-2)

---

## Overview
This repository contains a scalable machine learning framework for predicting student graduation outcomes in IT majors at UCAS, Gaza, Palestine. The full PySpark and scikit-learn implementation of the machine learning framework presented in the thesis above. It applies three classification algorithms — **Decision Tree (DT)**, **Random Forest (RF)**, and **Gradient Boosting Tree (GBT)** — to predict graduation outcomes for IT-major students at the University College of Applied Sciences (UCAS), Gaza, Palestine.

The dataset covers **19 years of student records (2000–2019)** across four IT specialisations:
- Programming & Databases
- Multimedia
- Web Design & Development
- Geographic Information Systems (GIS)


## Models

- Decision Tree (DT)
- Random Forest (RF)
- Gradient Boosting Tree (GBT)



---

## Key Results

| Algorithm | Accuracy (CA) | AUC | F1 | Precision | Recall | Spark Speed-up |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | **99.8%** | **0.999** | **0.998** | **0.998** | **0.998** | **75%** |
| Decision Tree | 97.1% | 0.979 | 0.971 | 0.972 | 0.971 | 58.9% |
| GBT Classifier | 93.0% | 0.965 | 0.923 | 0.931 | 0.930 | 72.9% |

**Top graduation predictors (Pearson correlation with STATUS):**

| # | Feature | r | Description |
|---|---------|---|-------------|
| 1 | `STD_AVG` | +0.997 | University cumulative GPA |
| 2 | `NORMAL_SMTR_COUNT` | +0.489 | Number of regular semesters |
| 3 | `SIFI_SMTR_COUNT` | +0.266 | Number of summer semesters |

---

## Repository Structure

```
UCAS-EDML-Graduation-Prediction/
├── data/
│   ├── sample_data.csv          # Anonymised 100-record sample
│   └── data_schema.md           # Full 21-attribute schema
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_models.ipynb
│   └── 04_spark_acceleration.ipynb
├── src/
│   ├── preprocessing.py         # Full data pipeline
│   ├── decision_tree.py         # DT — sklearn + PySpark
│   ├── random_forest.py         # RF — sklearn + PySpark
│   ├── gbt_classifier.py        # GBT — sklearn + PySpark
│   ├── spark_pipeline.py        # All 3 algorithms in one Spark job
│   └── evaluation.py            # Metrics, confusion matrix, ROC plots
├── results/
│   └── confusion_matrices/
├── docs/
│   └── spark_setup.md           # Cluster setup (Windows + Linux)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A — scikit-learn (no Spark required)

```bash
git clone https://github.com/sameredu/UCAS-EDML-Graduation-Prediction.git
cd UCAS-EDML-Graduation-Prediction
pip install -r requirements.txt

# Run best model (Random Forest)
python src/random_forest.py --data data/sample_data.csv

# Compare all three algorithms
python -c "
from src.preprocessing import UCASPreprocessor
from src.random_forest import run_sklearn_rf
from src.decision_tree import run_sklearn_dt
from src.gbt_classifier import run_sklearn_gbt

rf,  *_ = run_sklearn_rf('data/sample_data.csv')
dt,  *_ = run_sklearn_dt('data/sample_data.csv')
gbt, *_ = run_sklearn_gbt('data/sample_data.csv')
"
```

### Option B — Apache Spark (cluster)

```bash
# Start Master node
spark-class org.apache.spark.deploy.master.Master

# Start each Worker node (replace MASTER_IP)
spark-class org.apache.spark.deploy.worker.Worker spark://MASTER_IP:7077

# Submit full pipeline
spark-submit src/spark_pipeline.py \
    --master spark://MASTER_IP:7077 \
    --data data/ucas_data.csv

# Monitor jobs at: http://MASTER_IP:8080
```

See [`docs/spark_setup.md`](docs/spark_setup.md) for full cluster configuration.

---

## Dataset

The full UCAS dataset is subject to institutional data privacy restrictions and is not publicly distributed. An anonymised 100-record sample (`data/sample_data.csv`) is included for testing and demonstration.

- 21 features
- Binary target: Graduation status

Researchers wishing to access the full dataset should contact:

**University College of Applied Sciences (UCAS)**
Academic Affairs Office · Gaza, Palestine
[www.ucas.edu.ps](https://www.ucas.edu.ps)

### Schema (21 attributes)

| # | Attribute | Type | Description |
|---|-----------|------|-------------|
| 1 | `STUDY_TYPE_NAME` | Categorical | HS branch: 1=Literary, 2=Scientific, 3=Other |
| 2 | `DEPARTMENT_NAME` | Categorical | IT major: 1=Prog&DB, 2=Web, 3=Multimedia, 4=GIS |
| 3 | `STUDY_YEAR` | Int | Year of enrolment [2000–2019] |
| 4 | `STD_AVG` | Float | University cumulative GPA [35–100] |
| 5 | `STD_GRADEAVG` | Float | Graduate average [60–100] |
| 6 | `STD_ADDRESS` | Categorical | Region: 1=N.Gaza … 5=Rafah |
| 7 | `SEX` | Binary | 1=Male, 2=Female |
| 8 | `AGE` | Int | Student age [18–60] |
| 9 | `PERCENTAGE` | Float | High school average [50–100] |
| 10 | `TAW_YEAR` | Int | Year of HS diploma |
| 11 | `BIRTH_DATE` | Int | Year of birth |
| 12 | `NORMAL_SMTR_COUNT` | Int | Regular semesters [0–12] |
| 13 | `SIFI_SMTR_COUNT` | Int | Summer semesters [0–4] |
| 14 | `GRADUATE_YEAR` | Int | Graduation year (NULL if not graduated) |
| 15–16 | `YEAR_SMTR`, `STUDYPLAN_NO` | Int | Academic year / study plan ID |
| **17** | **`STATUS`** | **Binary** | **Target: 1=Graduated, 0=Not graduated** |

---



## Citation

@software{yaghi2026ucasedml,
  author = {Yaghi, Samer and Baraka, Rebhi S.},
  title = {UCAS Educational Big Data — ML Graduation Prediction},
  year = {2026},
  url = [{https://github.com/sameredu/UCAS-EDML-Graduation-Prediction}](https://github.com/sameredu/UCAS-EDML-Graduation-Prediction)
}

## License

MIT License


---

## Related Links

- 📄 ResearchGate: [researchgate.net/profile/Samer-Yaghi-2](https://www.researchgate.net/profile/Samer-Yaghi-2)
- 🔬 ORCID: [orcid.org/0009-0001-0268-7163](https://orcid.org/0009-0001-0268-7163)
- 🏛️ UCAS: [ucas.edu.ps](https://www.ucas.edu.ps)
- 🏛️ IUG: [iugaza.edu.ps](https://www.iugaza.edu.ps)

---

## License

This code is released under the [MIT License](LICENSE).
The dataset (sample only) is provided for academic research purposes.

---

*Faculty of Information Technology · University College of Applied Sciences (UCAS) · Islamic University of Gaza · Palestine · 2026*
