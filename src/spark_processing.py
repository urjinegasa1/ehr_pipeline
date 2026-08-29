"""
spark_processing.py
--------------------
Phase: PROCESSING (PySpark)

Demonstrates scalable processing of the patient encounter data using
PySpark, as required by the assignment ("PySpark for patient record
processing"). Reads the raw Parquet, performs distributed aggregations
and window-function analytics that would matter at real EHR scale
(millions of encounters across a hospital network), and writes a
Spark-native aggregate output.

Run standalone:  python3 spark_processing.py
Requires: Java 11+ and `pip install pyspark`
"""
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PARQUET = BASE_DIR / "data" / "raw" / "raw_encounters.parquet"
OUT_DIR = BASE_DIR / "data" / "processed" / "spark_aggregates"


def get_spark():
    return (
        SparkSession.builder
        .appName("ehr-pipeline-processing")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def run():
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(str(RAW_PARQUET))
    print(f"[SPARK] loaded {df.count():,} raw encounter rows, {len(df.columns)} columns")

    # --- Distributed cleaning: standardize categoricals & cast types ---
    df = df.withColumn("time_in_hospital", F.col("time_in_hospital").cast("double"))
    df = df.withColumn(
        "admission_type_clean",
        F.when(F.col("admission_type").isin("?", "", "Not Available"), "Unknown")
         .otherwise(F.col("admission_type"))
    )

    # --- Window function: patient-level encounter sequencing ---
    patient_window = Window.partitionBy("patient_id").orderBy("admit_datetime")
    df = df.withColumn("encounter_seq_for_patient", F.row_number().over(patient_window))
    df = df.withColumn(
        "days_since_prior_encounter",
        F.datediff(
            F.col("admit_datetime"),
            F.lag("admit_datetime", 1).over(patient_window)
        )
    )

    # --- Aggregation: readmission-relevant stats by admission_type & age bracket ---
    agg = (
        df.groupBy("admission_type_clean", "age")
        .agg(
            F.count("*").alias("n_encounters"),
            F.avg("time_in_hospital").alias("avg_length_of_stay"),
            F.avg("num_medications").alias("avg_num_medications"),
            F.avg("number_emergency").alias("avg_prior_er_visits"),
            F.countDistinct("patient_id").alias("n_unique_patients"),
        )
        .orderBy(F.desc("n_encounters"))
    )

    print("[SPARK] top admission_type x age_group aggregates:")
    agg.show(10, truncate=False)

    # --- Patients with rapid re-encounters (< 30 days apart): high-risk cohort ---
    rapid_reencounters = df.filter(F.col("days_since_prior_encounter") < 30).select(
        "patient_id", "encounter_id", "admit_datetime", "days_since_prior_encounter"
    )
    n_rapid = rapid_reencounters.count()
    print(f"[SPARK] identified {n_rapid:,} encounters that are rapid (<30d) re-admissions")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.coalesce(1).write.mode("overwrite").parquet(str(OUT_DIR / "admission_age_aggregates"))
    rapid_reencounters.coalesce(1).write.mode("overwrite").parquet(str(OUT_DIR / "rapid_reencounters"))
    print(f"[SPARK] wrote aggregate outputs -> {OUT_DIR}")

    spark.stop()


if __name__ == "__main__":
    run()
