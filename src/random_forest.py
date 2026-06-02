"""
random_forest.py
----------------
Random Forest classifier for UCAS graduation prediction.

Two implementations:
  (A) sklearn — for quick local experiments
  (B) PySpark MLlib — for big data / cluster execution (Appendix A.4)

Thesis results:
    CA = 99.8%,  AUC = 0.999,  Spark acceleration = 75%
    (60 s single-machine → 15 s with Spark, 16 jobs, 25 total tasks, 9.6 MB input)

Author : Samer Yaghi  |  syaghi@ucas.edu.ps  |  ORCID: 0009-0001-0268-7163
Supervisor : Prof. Rebhi S. Baraka  |  rbaraka@iugaza.edu.ps  |  Islamic University of Gaza
Institution: University College of Applied Sciences (UCAS), Gaza, Palestine
Thesis     : The Effect of Analyzing Big Educational Data on Students Performance (2022)
"""

import argparse
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import UCASPreprocessor
from src.evaluation import evaluate, plot_confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# (A)  sklearn Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def run_sklearn_rf(csv_path: str, n_estimators: int = 100,
                   save_cm: str = None) -> dict:
    """
    Train and evaluate a Random Forest using scikit-learn.

    Thesis configuration:
        - Multiple trees on bootstrapped subsets (bagging)
        - Majority vote for final prediction
        - CA = 99.8%, AUC = 0.999
    """
    prep = UCASPreprocessor(augment_factor=4)
    X_train, X_test, y_train, y_test = prep.run(csv_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_features="sqrt",       # Standard RF feature sampling
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    results = evaluate(y_test, y_pred, y_prob, model_name="Random Forest")

    if save_cm:
        plot_confusion_matrix(y_test, y_pred,
                              model_name="Random Forest",
                              save_path=save_cm)

    # ── Feature importance ────────────────────────────────────────────────────
    importances = sorted(
        zip(prep.feature_cols_used, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop feature importances:")
    for feat, imp in importances[:6]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    return results, rf, prep


# ══════════════════════════════════════════════════════════════════════════════
# (B)  PySpark MLlib Random Forest
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_rf(csv_path: str, master: str = "local[*]",
                 num_trees: int = 20) -> None:
    """
    Train Random Forest with Apache Spark MLlib.

    Matches Appendix A.4 cluster setup:
        - Master + 2 Workers
        - spark-class org.apache.spark.deploy.master.Master
        - spark-class org.apache.spark.deploy.worker.Worker spark://IP:7077
    Thesis timing: 16 jobs, 25 tasks, 9.6 MB input, 75% speed-up.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import RandomForestClassifier as SparkRF
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
                                          MulticlassClassificationEvaluator
        import time
    except ImportError:
        print("[spark] PySpark not installed. Run: pip install pyspark")
        return

    spark = (SparkSession.builder
             .appName("UCAS-RF-Classifier")
             .master(master)
             .config("spark.sql.shuffle.partitions", "8")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df = df.na.drop(subset=["STATUS"])

    feature_cols = [c for c in df.columns if c != "STATUS"]
    assembler = VectorAssembler(inputCols=feature_cols,
                                outputCol="features",
                                handleInvalid="keep")

    rf = SparkRF(
        labelCol="STATUS",
        featuresCol="features",
        numTrees=num_trees,
        maxDepth=8,
        featureSubsetStrategy="sqrt",
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, rf])
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    print(f"[spark-RF] Training on {train_df.count():,} records, "
          f"{num_trees} trees…")
    t0 = time.time()
    model = pipeline.fit(train_df)
    elapsed = time.time() - t0
    print(f"[spark-RF] Training time: {elapsed:.1f}s")

    predictions = model.transform(test_df)

    auc = BinaryClassificationEvaluator(
        labelCol="STATUS", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    ca = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    print(f"\n[spark-RF] Results:")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {ca*100:.2f}%")
    print(f"  Training : {elapsed:.1f}s  (thesis single-machine baseline: 60s)")

    # Feature importance from the trained RF model
    rf_model = model.stages[-1]
    print("\n  Feature importances (top 5):")
    importances = sorted(
        enumerate(rf_model.featureImportances.toArray()),
        key=lambda x: x[1], reverse=True
    )[:5]
    for idx, imp in importances:
        name = feature_cols[idx] if idx < len(feature_cols) else f"feature_{idx}"
        print(f"    {name:<25} {imp:.4f}")

    spark.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCAS Random Forest")
    parser.add_argument("--data",          default="data/sample_data.csv")
    parser.add_argument("--n-estimators",  type=int, default=100)
    parser.add_argument("--spark",         action="store_true")
    parser.add_argument("--master",        default="local[*]")
    parser.add_argument("--save-cm",       default=None)
    args = parser.parse_args()

    if args.spark:
        run_spark_rf(args.data, master=args.master,
                     num_trees=args.n_estimators)
    else:
        run_sklearn_rf(args.data, n_estimators=args.n_estimators,
                       save_cm=args.save_cm)
