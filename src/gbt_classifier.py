"""
gbt_classifier.py
-----------------
Gradient Boosting Tree classifier for UCAS graduation prediction.

Two implementations:
  (A) sklearn — for quick local experiments
  (B) PySpark MLlib — for big data / cluster execution (Appendix A.3)

Thesis results:
    CA = 93.0%,  AUC = 0.965,  Spark acceleration = 72.9%
    (150 jobs, 257 tasks, 97 MB input, 1.6 min → 26 s with Spark)

Author : Samer A. Yaghi (The Islamic University of Gaza, 2022)
"""

import argparse
from sklearn.ensemble import GradientBoostingClassifier
from src.preprocessing import UCASPreprocessor
from src.evaluation import evaluate, plot_confusion_matrix


# ══════════════════════════════════════════════════════════════════════════════
# (A)  sklearn GBT
# ══════════════════════════════════════════════════════════════════════════════

def run_sklearn_gbt(csv_path: str, n_estimators: int = 100,
                    learning_rate: float = 0.1, save_cm: str = None) -> dict:
    """
    Train and evaluate a GBT using scikit-learn.

    Gradient Boosting builds trees sequentially:
        F_m(x) = F_{m-1}(x) + η * h_m(x)
    where h_m corrects the residuals of F_{m-1}.

    Thesis description (Section 4.2.3):
        Step 1: Calculate mean of target.
        Step 2: Calculate residuals.
        Step 3: Build decision tree on residuals.
        Step 4: Predict using all trees.
        Steps 5-7: Iterate until n_estimators.
    """
    prep = UCASPreprocessor(augment_factor=4)
    X_train, X_test, y_train, y_test = prep.run(csv_path)

    gbt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=3,              # Shallow trees (weak learners)
        subsample=0.8,
        random_state=42,
    )
    gbt.fit(X_train, y_train)

    y_pred = gbt.predict(X_test)
    y_prob = gbt.predict_proba(X_test)[:, 1]

    results = evaluate(y_test, y_pred, y_prob, model_name="GBT Classifier")

    if save_cm:
        plot_confusion_matrix(y_test, y_pred,
                              model_name="GBT Classifier",
                              save_path=save_cm)
    return results, gbt, prep


# ══════════════════════════════════════════════════════════════════════════════
# (B)  PySpark MLlib GBT
# ══════════════════════════════════════════════════════════════════════════════

def run_spark_gbt(csv_path: str, master: str = "local[*]",
                  max_iter: int = 20) -> None:
    """
    Train GBT with Apache Spark MLlib.

    Matches Appendix A.3:
        Thesis timing: 150 jobs, 257 tasks, 97 MB input, 72.9% speed-up.
    Note: GBT is more resource-intensive than RF/DT due to sequential boosting.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.ml import Pipeline
        from pyspark.ml.classification import GBTClassifier as SparkGBT
        from pyspark.ml.feature import VectorAssembler
        from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
                                          MulticlassClassificationEvaluator
        import time
    except ImportError:
        print("[spark] PySpark not installed. Run: pip install pyspark")
        return

    spark = (SparkSession.builder
             .appName("UCAS-GBT-Classifier")
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

    gbt = SparkGBT(
        labelCol="STATUS",
        featuresCol="features",
        maxIter=max_iter,
        maxDepth=3,
        stepSize=0.1,          # learning rate
        subsamplingRate=0.8,
        seed=42,
    )

    pipeline = Pipeline(stages=[assembler, gbt])
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

    print(f"[spark-GBT] Training on {train_df.count():,} records, "
          f"{max_iter} iterations…")
    print("  Note: GBT is sequential — expect longer training than RF/DT.")
    t0 = time.time()
    model = pipeline.fit(train_df)
    elapsed = time.time() - t0
    print(f"[spark-GBT] Training time: {elapsed:.1f}s "
          f"(thesis single-machine baseline: ~96s)")

    predictions = model.transform(test_df)

    auc = BinaryClassificationEvaluator(
        labelCol="STATUS", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    ca = MulticlassClassificationEvaluator(
        labelCol="STATUS", predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    print(f"\n[spark-GBT] Results:")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Accuracy : {ca*100:.2f}%")

    spark.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCAS GBT Classifier")
    parser.add_argument("--data",          default="data/sample_data.csv")
    parser.add_argument("--n-estimators",  type=int, default=100)
    parser.add_argument("--lr",            type=float, default=0.1,
                        help="Learning rate (sklearn only)")
    parser.add_argument("--spark",         action="store_true")
    parser.add_argument("--master",        default="local[*]")
    parser.add_argument("--save-cm",       default=None)
    args = parser.parse_args()

    if args.spark:
        run_spark_gbt(args.data, master=args.master,
                      max_iter=args.n_estimators)
    else:
        run_sklearn_gbt(args.data, n_estimators=args.n_estimators,
                        learning_rate=args.lr, save_cm=args.save_cm)
