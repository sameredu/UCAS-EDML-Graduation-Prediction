# Spark Cluster Setup Guide

Replicates the environment described in Appendix A of the thesis.

## Specifications (thesis cluster)

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7-8550U @ 1.80 GHz (4 cores) |
| RAM | 2.3 GB |
| OS | Windows 8.1 Pro 64-bit |
| Python | 3.7.2 |
| Java | JDK 8u192 |
| Topology | 1 Master + 2 Workers |

## Installation

```bash
# 1. Install Java JDK 8+
# 2. Install Python 3.7+
pip install pyspark

# 3. Start Master
spark-class org.apache.spark.deploy.master.Master

# 4. Start each Worker (replace MASTER_IP)
spark-class org.apache.spark.deploy.worker.Worker spark://MASTER_IP:7077

# 5. Monitor at http://MASTER_IP:8080
```

## Run the pipeline

```bash
spark-submit src/spark_pipeline.py --master spark://MASTER_IP:7077 --data data/ucas_data.csv
```

## Expected Spark acceleration (from thesis Table 5.5)

| Algorithm | Jobs | Tasks | Single-machine | Spark time | Acceleration |
|-----------|------|-------|---------------|------------|-------------|
| Random Forest | 16 | 25 | 60 s | 15 s | **75%** |
| GBT Classifier | 150 | 257 | 1.6 min | 26 s | 72.9% |
| Decision Tree | 11 | 16 | 39 s | 16 s | 58.9% |
