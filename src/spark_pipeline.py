"""
spark_pipeline.py
-----------------
Full Apache Spark ML pipeline — runs DT, RF, and GBT on UCAS data
and prints a comparative performance table.

Usage:
    # Local mode (development)
    python src/spark_pipeline.py --data data/ucas_data.csv

    # Cluster mode (matches thesis Appendix A setup)
    spark-submit src/spark_pipeline.py \\
        --master spark://192.168.1.100:7077 \\
        --data data/ucas_data.csv

Cluster setup (Windows, matching thesis):
    # On Master machine:
    spark-class org.apache.spark.deploy.master.Master

    # On each Worker machine:
    spark-class org.apache.spark.deploy.worker.Worker spark://MASTER_IP:7077

    # Monitor at: http://MASTER_IP:8080

Author : Samer Yaghi  |  syaghi@ucas.edu.ps  |  ORCID: 0009-0001-0268-7163
Supervisor : Prof. Rebhi S. Baraka  |  rbaraka@iugaza.edu.ps  |  Islamic University of Gaza
Institution: University College of Applied Sciences (UCAS), Gaza, Palestine
Thesis     : The Effect of Analyzing Big Educational Data on Students Performance (2022)
"""

import argparse
import time

# ── PySpark imports (guarded) ─────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import (
        DecisionTreeClassifier,
        RandomForestClassifier,
        GBTClassifier,
    )
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator,
    )
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False


def build_spark_session(master: str, app_name: str) -> "SparkSession":
    return (SparkSession.builder
            .appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.executor.memory", "1g")
            .getOrCreate())


def load_and_prepare(spark, csv_path: str):
    """Load CSV and build feature vector."""
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.na.drop(subset=["STATUS"])
    feature_cols = [c for c in df.columns if c != "STATUS"]
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )
    df_vec = assembler.transform(df).select("features", "STATUS")
    return df_vec, feature_cols


def make_evaluators():
    auc_eval = BinaryClassificationEvaluator(
        labelCol="STATUS", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    ca_eval = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="f1"
    )
    prec_eval = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="weightedPrecision"
    )
    rec_eval = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="weightedRecall"
    )
    return auc_eval, ca_eval, f1_eval, prec_eval, rec_eval


def run_algorithm(name, classifier, train_df, test_df, evaluators):
    """Fit one classifier, time it, and return metrics dict."""
    auc_e, ca_e, f1_e, prec_e, rec_e = evaluators

    print(f"\n[Spark] Training {name}…")
    t0 = time.time()
    model = classifier.fit(train_df)
    elapsed = time.time() - t0

    preds = model.transform(test_df)

    return {
        "Algorithm":   name,
        "CA (%)":      round(ca_e.evaluate(preds) * 100, 2),
        "AUC":         round(auc_e.evaluate(preds), 4),
        "F1":          round(f1_e.evaluate(preds), 4),
        "Precision":   round(prec_e.evaluate(preds), 4),
        "Recall":      round(rec_e.evaluate(preds), 4),
        "Train time":  f"{elapsed:.1f}s",
    }


def print_results_table(results: list):
    """Pretty-print a comparison table matching Table 5.4 of the thesis."""
    cols = ["Algorithm", "CA (%)", "AUC", "F1", "Precision", "Recall", "Train time"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in results))
              for c in cols}
    sep = "+-" + "-+-".join("-" * widths[c] for c in cols) + "-+"
    header = "| " + " | ".join(f"{c:<{widths[c]}}" for c in cols) + " |"

    print("\n" + "=" * 60)
    print("  UCAS ML Model Comparison (Apache Spark)")
    print("=" * 60)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        row = "| " + " | ".join(f"{str(r.get(c,'')):<{widths[c]}}" for c in cols) + " |"
        print(row)
    print(sep)
    best = max(results, key=lambda x: x["CA (%)"])
    print(f"\n  Best model: {best['Algorithm']}  (CA = {best['CA (%)']}%)")
    print()


def main(csv_path: str, master: str):
    if not PYSPARK_AVAILABLE:
        print("PySpark is not installed. Install it with:\n  pip install pyspark")
        return

    spark = build_spark_session(master, "UCAS-ML-Pipeline")
    spark.sparkContext.setLogLevel("WARN")

    print(f"[Spark] Session started. Master: {master}")
    print(f"[Spark] Loading data from: {csv_path}")

    df_vec, feature_cols = load_and_prepare(spark, csv_path)
    train_df, test_df = df_vec.randomSplit([0.7, 0.3], seed=42)

    print(f"[Spark] Train: {train_df.count():,}  |  Test: {test_df.count():,}")
    print(f"[Spark] Features: {len(feature_cols)}")

    evaluators = make_evaluators()

    # ── Define classifiers ────────────────────────────────────────────────────
    classifiers = [
        ("Decision Tree", DecisionTreeClassifier(
            labelCol="STATUS", featuresCol="features",
            maxDepth=6, impurity="gini", seed=42
        )),
        ("Random Forest", RandomForestClassifier(
            labelCol="STATUS", featuresCol="features",
            numTrees=20, maxDepth=8,
            featureSubsetStrategy="sqrt", seed=42
        )),
        ("GBT Classifier", GBTClassifier(
            labelCol="STATUS", featuresCol="features",
            maxIter=20, maxDepth=3, stepSize=0.1,
            subsamplingRate=0.8, seed=42
        )),
    ]

    # ── Run all classifiers ───────────────────────────────────────────────────
    results = []
    for name, clf in classifiers:
        metrics = run_algorithm(name, clf, train_df, test_df, evaluators)
        results.append(metrics)

    print_results_table(results)

    # ── Expected results from thesis (for reference) ──────────────────────────
    print("  Thesis reported results:")
    print("    Random Forest : CA=99.8%, AUC=0.999")
    print("    Decision Tree : CA=97.1%, AUC=0.979")
    print("    GBT Classifier: CA=93.0%, AUC=0.965")
    print()

    spark.stop()
    print("[Spark] Session stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCAS Spark ML Pipeline")
    parser.add_argument("--data",   default="data/sample_data.csv")
    parser.add_argument("--master", default="local[*]",
                        help="Spark master URL (local[*] or spark://IP:7077)")
    args = parser.parse_args()
    main(args.data, args.master)
