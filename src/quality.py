"""
quality.py
----------
Phase: DATA QUALITY (Great Expectations)

Defines and runs a validation suite against the ML-ready EHR dataset,
covering the assignment's "medical data validation" requirements:
  - range checks (clinical vitals within plausible bounds)
  - completeness / missing-rate checks
  - categorical domain checks (ICD categories, admission types, etc.)
  - uniqueness (encounter_id primary key)
  - consistency checks (target is binary, no negative counts)

Produces a JSON data-quality report at docs/data_quality_report.json and
prints a human-readable summary.
"""
import json
from pathlib import Path

import great_expectations as gx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "ehr_ml_ready.parquet"
REPORT_PATH = BASE_DIR / "docs" / "data_quality_report.json"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_and_run_suite(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("ehr_pandas_source")
    data_asset = data_source.add_dataframe_asset(name="ehr_ml_ready_asset")
    batch_def = data_asset.add_batch_definition_whole_dataframe("ehr_ml_ready_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="ehr_ml_ready_suite")

    expectations = [
        # --- Uniqueness / primary key ---
        gx.expectations.ExpectColumnValuesToBeUnique(column="encounter_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="encounter_id"),

        # --- Completeness ---
        gx.expectations.ExpectColumnValuesToNotBeNull(column="readmitted_within_30d"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="time_in_hospital"),

        # --- Range checks (clinical plausibility) ---
        gx.expectations.ExpectColumnValuesToBeBetween(column="time_in_hospital", min_value=0, max_value=60),
        gx.expectations.ExpectColumnValuesToBeBetween(column="num_medications", min_value=0, max_value=100),
        gx.expectations.ExpectColumnValuesToBeBetween(column="bmi", min_value=10, max_value=70),
        gx.expectations.ExpectColumnValuesToBeBetween(column="systolic_bp", min_value=50, max_value=250),
        gx.expectations.ExpectColumnValuesToBeBetween(column="diastolic_bp", min_value=30, max_value=150),
        gx.expectations.ExpectColumnValuesToBeBetween(column="creatinine", min_value=0, max_value=15),
        gx.expectations.ExpectColumnValuesToBeBetween(column="hba1c_value", min_value=2, max_value=20),

        # --- No negative counts ---
        gx.expectations.ExpectColumnValuesToBeBetween(column="number_outpatient", min_value=0),
        gx.expectations.ExpectColumnValuesToBeBetween(column="number_emergency", min_value=0),
        gx.expectations.ExpectColumnValuesToBeBetween(column="number_inpatient", min_value=0),

        # --- Categorical domain checks ---
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="admission_type",
            value_set=["Emergency", "Urgent", "Elective", "Newborn", "Not Available", "Trauma Center"],
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="readmitted_within_30d", value_set=[0, 1]
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="diag_1_category",
            value_set=["Diabetes", "Circulatory", "Respiratory", "Genitourinary",
                       "Other_Metabolic_Symptom", "Administrative_Followup",
                       "Other_ICD9", "Other_ICD10", "Unknown"],
        ),

        # --- Missing-rate check (flagging, not hard-failing, high-missingness cols) ---
        gx.expectations.ExpectColumnValuesToNotBeNull(column="bmi_category", mostly=0.95),
    ]
    for exp in expectations:
        suite.add_expectation(exp)

    context.suites.add(suite)
    validation_result = batch.validate(suite)
    return validation_result


def summarize_and_save(validation_result):
    results = validation_result.to_json_dict()
    n_total = len(results["results"])
    n_success = sum(1 for r in results["results"] if r["success"])

    summary = {
        "overall_success": results["success"],
        "expectations_total": n_total,
        "expectations_passed": n_success,
        "expectations_failed": n_total - n_success,
        "failed_expectations": [
            {
                "expectation_type": r["expectation_config"]["type"],
                "column": r["expectation_config"]["kwargs"].get("column"),
                "unexpected_count": r["result"].get("unexpected_count"),
                "unexpected_percent": r["result"].get("unexpected_percent"),
            }
            for r in results["results"] if not r["success"]
        ],
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[QUALITY] {n_success}/{n_total} expectations passed")
    if summary["failed_expectations"]:
        print("[QUALITY] Failed expectations:")
        for f_exp in summary["failed_expectations"]:
            print(f"   - {f_exp['expectation_type']} on '{f_exp['column']}': "
                  f"{f_exp['unexpected_count']} unexpected ({f_exp['unexpected_percent']:.2f}%)")
    print(f"[QUALITY] report saved -> {REPORT_PATH}")
    return summary


def run():
    df = pd.read_parquet(DATA_PATH)
    result = build_and_run_suite(df)
    return summarize_and_save(result)


if __name__ == "__main__":
    run()
