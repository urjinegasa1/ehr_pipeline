"""
features.py
-----------
Phase: FEATURE ENGINEERING

Creates 15 domain-relevant features (>= 10 required) for the readmission
prediction task, and produces the final ML-ready dataset.

Feature dictionary (rationale for each is documented inline and repeated
in docs/data_dictionary.md):
  1.  length_of_stay_bucket    - categorical LOS bucket, non-linear risk signal
  2.  age_group_numeric        - ordinal encoding of the age bracket midpoint
  3.  comorbidity_index        - count of distinct diagnosis categories present
  4.  diagnosis_complexity     - weighted score combining #diagnoses & #procedures
  5.  polypharmacy_flag        - >=10 medications, known readmission risk factor
  6.  prior_utilization_score  - weighted sum of past outpatient/ER/inpatient visits
  7.  is_emergency_admission   - binary flag from admission_type
  8.  had_recent_er_visit      - binary flag, number_emergency > 0
  9.  abnormal_a1c_flag        - A1C result was '>7' or '>8'
  10. abnormal_glucose_flag    - glucose serum result was '>200' or '>300'
  11. bmi_category             - clinical BMI banding (underweight..obese)
  12. hypertension_flag        - systolic/diastolic over clinical thresholds
  13. renal_risk_flag          - creatinine above normal range
  14. lab_completeness_score   - fraction of lab fields present (data-quality-aware feature)
  15. readmission_risk_score   - composite weighted risk score (business feature)
  16. diabetes_med_changed     - medication changed AND patient is on diabetes meds
"""
from pathlib import Path

import numpy as np
import pandas as pd

from storage import write_stage, get_conn

AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
    "Unknown": np.nan,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. length_of_stay_bucket
    df["length_of_stay_bucket"] = pd.cut(
        df["time_in_hospital"], bins=[0, 2, 5, 10, np.inf],
        labels=["short(1-2d)", "medium(3-5d)", "long(6-10d)", "extended(10d+)"]
    ).astype("string")

    # 2. age_group_numeric
    df["age_group_numeric"] = df["age"].map(AGE_MIDPOINT)

    # 3. comorbidity_index: distinct non-Unknown diagnosis categories across diag_1..3
    diag_cat_cols = ["diag_1_category", "diag_2_category", "diag_3_category"]
    df["comorbidity_index"] = df[diag_cat_cols].apply(
        lambda row: len({v for v in row if pd.notna(v) and v != "Unknown"}), axis=1
    )

    # 4. diagnosis_complexity: weighted combination of diagnosis count & procedures
    df["diagnosis_complexity"] = (
        0.5 * df["number_diagnoses"] + 0.3 * df["num_procedures"] + 0.2 * df["comorbidity_index"]
    ).round(2)

    # 5. polypharmacy_flag
    df["polypharmacy_flag"] = (df["num_medications"] >= 10).astype(int)

    # 6. prior_utilization_score
    df["prior_utilization_score"] = (
        1.0 * df["number_outpatient"] + 2.0 * df["number_emergency"] + 3.0 * df["number_inpatient"]
    )

    # 7. is_emergency_admission
    df["is_emergency_admission"] = (df["admission_type"] == "Emergency").astype(int)

    # 8. had_recent_er_visit
    df["had_recent_er_visit"] = (df["number_emergency"] > 0).astype(int)

    # 9. abnormal_a1c_flag
    df["abnormal_a1c_flag"] = df["A1Cresult"].isin([">7", ">8"]).astype(int)

    # 10. abnormal_glucose_flag
    df["abnormal_glucose_flag"] = df["max_glu_serum"].isin([">200", ">300"]).astype(int)

    # 11. bmi_category
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"]
    ).astype("string")

    # 12. hypertension_flag (clinical thresholds: >=140/90)
    df["hypertension_flag"] = ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)

    # 13. renal_risk_flag (creatinine > 1.3 mg/dL commonly flags reduced renal function)
    df["renal_risk_flag"] = (df["creatinine"] > 1.3).astype(int)

    # 14. lab_completeness_score: fraction of key lab fields present (data-quality feature)
    lab_fields = ["hba1c_value", "ldl_cholesterol", "creatinine", "bmi", "systolic_bp", "diastolic_bp"]
    df["lab_completeness_score"] = df[lab_fields].notna().mean(axis=1).round(2)

    # 15. readmission_risk_score: composite weighted business risk score
    df["readmission_risk_score"] = (
        0.20 * df["prior_utilization_score"]
        + 0.15 * df["comorbidity_index"]
        + 0.15 * df["polypharmacy_flag"]
        + 0.15 * df["abnormal_a1c_flag"]
        + 0.10 * df["hypertension_flag"]
        + 0.10 * df["renal_risk_flag"]
        + 0.15 * (df["length_of_stay_bucket"].isin(["long(6-10d)", "extended(10d+)"])).astype(int)
    ).round(3)

    # 16. diabetes_med_changed
    df["diabetes_med_changed"] = ((df["change"] == "Ch") & (df["diabetesMed"] == "Yes")).astype(int)

    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Binary target per assignment spec: readmission (binary)
    df["readmitted_within_30d"] = (df["readmitted"] == "<30").astype(int)
    return df


def build_ml_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Drops leakage / PII-adjacent columns; keeps engineered features + target."""
    drop_cols = [
        "patient_id",  # pseudonymized identifier, not a feature -> privacy-aware drop
        "readmitted",  # raw source of the target, would leak
        "admit_datetime", "discharge_datetime",  # raw timestamps (info captured in LOS features)
        "diag_1", "diag_2", "diag_3",  # replaced by *_category features
        "lab_source_system", "retrieved_at",  # source metadata, not clinical signal
    ]
    ml_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return ml_df


def run():
    with get_conn() as conn:
        merged = pd.read_sql("SELECT * FROM interim_merged_clean", conn)

    featured = engineer_features(merged)
    featured = build_target(featured)

    n_new_features = 16
    print(f"[FEATURES] engineered {n_new_features} new features; shape now {featured.shape}")

    write_stage(featured, "interim", "encounters_featured")

    ml_ready = build_ml_ready(featured)
    write_stage(ml_ready, "processed", "ehr_ml_ready")

    # Also emit a plain CSV for the 3rd required storage format / easy inspection
    csv_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "ehr_ml_ready.csv"
    ml_ready.to_csv(csv_path, index=False)
    print(f"[FEATURES] wrote ML-ready CSV -> {csv_path}")

    print(f"[FEATURES] target balance:\n{ml_ready['readmitted_within_30d'].value_counts(normalize=True)}")
    return ml_ready


if __name__ == "__main__":
    run()
