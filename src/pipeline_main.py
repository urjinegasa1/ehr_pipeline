"""
pipeline_main.py
-----------------
Runs the full pipeline end-to-end outside of Airflow, for local development
and for the demo recording:

    extract (generate_raw_sources) -> storage -> transform -> features
    -> quality (Great Expectations) -> spark_processing

This mirrors the Airflow DAG task order exactly (dags/ehr_pipeline_dag.py)
so behavior is identical whether orchestrated by Airflow or run manually.
"""
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("Ingestion (generate raw sources)", "generate_raw_sources.py"),
    ("Storage (SQLite + Parquet)", "storage.py"),
    ("Cleaning / Transformation", "transform.py"),
    ("Feature Engineering", "features.py"),
    ("Data Quality (Great Expectations)", "quality.py"),
    ("Distributed Processing (PySpark)", "spark_processing.py"),
]


def main():
    src_dir = Path(__file__).resolve().parent
    for label, script in STEPS:
        print(f"\n{'='*70}\nSTEP: {label}\n{'='*70}")
        start = time.time()
        result = subprocess.run([sys.executable, script], cwd=src_dir)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"[PIPELINE] FAILED at step '{label}' after {elapsed:.1f}s")
            sys.exit(1)
        print(f"[PIPELINE] completed '{label}' in {elapsed:.1f}s")
    print("\nPipeline completed successfully. ML-ready dataset at "
          "data/processed/ehr_ml_ready.{parquet,csv} and SQLite table processed_ehr_ml_ready.")


if __name__ == "__main__":
    main()
