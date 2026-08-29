"""
ehr_pipeline_dag.py
--------------------
Phase: ORCHESTRATION (Apache Airflow)

DAG: extract -> transform -> engineer_features -> validate -> spark_process -> load

Design notes (documented for the report's Orchestration section):
  - Each stage is an independent, idempotent task calling into src/*.py so
    the same code runs identically whether triggered by Airflow or run
    manually for development (see README "Running the pipeline manually").
  - Error handling: `retries=2` with exponential backoff on every task;
    a dedicated `on_failure_callback` posts a structured alert (logged
    here; swap in Slack/email/SNS in production).
  - Privacy-aware processing: the `validate` task fails the DAG (no
    downstream run) if the Great Expectations suite does not pass, so
    malformed or leaking PHI never reaches the ML-ready output.
  - Schedule: daily at 02:00, mimicking a nightly EHR batch ETL window.

To run for real: copy this file into your local $AIRFLOW_HOME/dags/,
`pip install apache-airflow`, `airflow db init`, `airflow webserver` +
`airflow scheduler`, then trigger `ehr_healthcare_pipeline` from the UI
or `airflow dags trigger ehr_healthcare_pipeline`.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import sys
from pathlib import Path

# Make src/ importable when Airflow loads this DAG file
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _log_failure(context):
    ti = context["task_instance"]
    print(f"[ALERT] Task {ti.task_id} failed on run {context['run_id']}. "
          f"Exception: {context.get('exception')}. "
          f"(Wire this to Slack/PagerDuty/SNS in production.)")


default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "on_failure_callback": _log_failure,
}

with DAG(
    dag_id="ehr_healthcare_pipeline",
    description="End-to-end EHR readmission-risk data pipeline (Topic 4)",
    default_args=default_args,
    schedule="0 2 * * *",       # nightly at 02:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["healthcare", "ehr", "data-engineering-assessment"],
) as dag:

    def task_extract(**_):
        from generate_raw_sources import build_encounters, build_lab_api_extract, RAW_DIR
        import json
        df = build_encounters()
        df.to_csv(RAW_DIR / "ehr_encounters.csv", index=False)
        labs = build_lab_api_extract(df["encounter_id"].unique().tolist())
        with open(RAW_DIR / "ehr_lab_api_extract.json", "w") as f:
            json.dump(labs, f)
        print(f"Extracted {len(df)} encounters, {len(labs)} lab records")

    def task_storage(**_):
        from storage import load_raw_to_sqlite_and_parquet
        load_raw_to_sqlite_and_parquet()

    def task_transform(**_):
        from transform import run as transform_run
        transform_run()

    def task_engineer_features(**_):
        from features import run as features_run
        features_run()

    def task_validate(**_):
        from quality import run as quality_run
        summary = quality_run()
        if not summary["overall_success"]:
            raise ValueError(
                f"Data quality validation FAILED: "
                f"{summary['expectations_failed']} expectation(s) did not pass. "
                f"Halting DAG to prevent bad data reaching the ML-ready output."
            )

    def task_spark_process(**_):
        from spark_processing import run as spark_run
        spark_run()

    def task_load_final(**_):
        import shutil
        from pathlib import Path
        base = Path(__file__).resolve().parents[1]
        src = base / "data" / "processed" / "ehr_ml_ready.parquet"
        dst = base / "data" / "processed" / "ehr_ml_ready_LATEST.parquet"
        shutil.copy(src, dst)
        print(f"Published ML-ready dataset -> {dst}")

    extract = PythonOperator(task_id="extract", python_callable=task_extract)
    store_raw = PythonOperator(task_id="store_raw", python_callable=task_storage)
    transform = PythonOperator(task_id="transform_clean", python_callable=task_transform)
    engineer = PythonOperator(task_id="engineer_features", python_callable=task_engineer_features)
    validate = PythonOperator(task_id="validate_quality", python_callable=task_validate)
    spark_process = PythonOperator(task_id="spark_process", python_callable=task_spark_process)
    load_final = PythonOperator(task_id="load_final", python_callable=task_load_final)

    extract >> store_raw >> transform >> engineer >> validate >> spark_process >> load_final
