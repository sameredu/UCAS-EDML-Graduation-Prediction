"""
decision_tree.py
----------------
Decision Tree classifier for UCAS graduation prediction.

Two implementations:
  (A) sklearn — for quick local experiments
  (B) PySpark MLlib — for big data / cluster execution

Algorithm steps match Section 4.2.1 of the thesis.

Author : Samer A. Yaghi (The Islamic University of Gaza, 2022)
"""

import argparse
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from src.preprocessing import UCASPreprocessor
from src.evaluation import evaluate, plot_confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# (A)  sklearn Decision Tree
# ══════════════════════════════════════════════════════════════════════════════

def run_sklearn_dt(csv_path: str, max_depth: int = None,
                   save_cm: str = None) -> dict:
    """
    Train and evaluate a Decision Tree using scikit-learn.

    The thesis DT produced:
        - 923 nodes, 462 leaves at depth 6
        - CA = 97.1%,  AUC ≈ 0.979
    """
    prep = UCASPreprocessor(augment_factor=4)
    X_train, X_test, y_train, y_test = prep.run(csv_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(
        criterion="gini",          # Gini index (ASM used in thesis)
        max_depth=max_depth,       # None = full tree; set 6 to replicate thesis
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    dt.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = dt.predict(X_test)
    y_prob = dt.predict_proba(X_test)[:, 1]

    results = evaluate(y_test, y_pred, y_prob, model_name="Decision Tree")

    if save_cm:
        plot_confusion_matrix(y_test, y_pred,
                              model_name="Decision Tree",
                              save_path=save_cm)

    # ── Print decision rules (first 3 levels) ─────────────────────────────────
    print("Decision rules (first 3 levels):")
    print(export_text(dt,
                      feature_names=prep.feature_cols_used,
                      max_depth=3))

    return results, dt, prep


# ══════════════════════════════════════════════════════════════════════════════
# (B)  PySpark MLlib Decision Tree
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_dt(csv_path: str, master: str = "local[*]") -> None:
    """
    Train a Decision Tree using Apache Spark MLlib.

    Matches the Spark cluster setup described in Appendix A.2 of the thesis:
        spark-submit src/decision_tree.py --spark --data data/ucas_data.csv
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import DecisionTreeClassifier as SparkDT
        from pyspark.ml.feature import VectorAssembler, StringIndexer
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
                                          MulticlassClassificationEvaluator
        import time
    except ImportError:
        print("[spark] PySpark not installed. Run: pip install pyspark")
        return

    spark = (SparkSession.builder
             .appName("UCAS-DT-Classifier")
             .master(master)
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.na.drop(subset=["STATUS"])

    feature_cols = [c for c in df.columns if c != "STATUS"]

    # ── Pipeline: Assemble features → DT ─────────────────────────────────────
    assembler = VectorAssembler(inputCols=feature_cols,
                                outputCol="features",
                                handleInvalid="keep")

    dt = SparkDT(
        labelCol="STATUS",
        featuresCol="features",
        maxDepth=6,           # Matches thesis tree depth
        impurity="gini",
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, dt])

    # ── Train / test split (70 / 30) ─────────────────────────────────────────
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    print(f"[spark-DT] Training on {train_df.count():,} records…")
    t0 = time.time()
    model = pipeline.fit(train_df)
    elapsed = time.time() - t0
    print(f"[spark-DT] Training time: {elapsed:.1f}s")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    predictions = model.transform(test_df)

    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="STATUS", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_ca = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="accuracy"
    )

    auc = evaluator_auc.evaluate(predictions)
    ca  = evaluator_ca.evaluate(predictions)

    print(f"\n[spark-DT] Results:")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {ca*100:.2f}%")

    spark.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCAS Decision Tree")
    parser.add_argument("--data",     default="data/sample_data.csv")
    parser.add_argument("--depth",    type=int, default=None,
                        help="Max tree depth (default: unlimited)")
    parser.add_argument("--spark",    action="store_true",
                        help="Use PySpark MLlib instead of scikit-learn")
    parser.add_argument("--master",   default="local[*]",
                        help="Spark master URL (e.g. spark://192.168.1.1:7077)")
    parser.add_argument("--save-cm",  default=None,
                        help="Path to save confusion matrix image")
    args = parser.parse_args()

    if args.spark:
        run_spark_dt(args.data, master=args.master)
    else:
        run_sklearn_dt(args.data, max_depth=args.depth, save_cm=args.save_cm)
