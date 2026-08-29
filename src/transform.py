"""
transform.py
------------
Phase: CLEANING / TRANSFORMATION

Tasks (per assignment spec for Topic 4 - Healthcare/EHR):
  - Handle missing values
  - Standardize ICD-9 / ICD-10 diagnosis codes into a unified taxonomy
  - Validate/parse admit & discharge dates
  - Standardize categorical text
  - Remove duplicate records
  - Join the two raw sources (encounters + labs) on encounter_id
  - Basic privacy-aware processing: drop direct patient free-text fields
    (none present here, but pseudonymized patient_id is retained only as a
    join key and dropped from the final ML-ready output in features.py)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from storage import load_raw_to_sqlite_and_parquet, write_stage

# ---- ICD standardization map --------------------------------------------
# Real pipelines would use a full ICD-9<->ICD-10 GEM crosswalk; here we map
# both coding systems into a small set of clinically meaningful categories,
# which is what's actually useful for the downstream ML task.
def categorize_icd(code):
    if pd.isna(code) or code in ("", "?", None):
        return "Unknown"
    code = str(code).strip().upper()

    # ICD-10 style (starts with a letter)
    if re.match(r"^[A-Z]", code):
        if code.startswith("E11") or code.startswith("E10"):
            return "Diabetes"
        if code.startswith("I1") or code.startswith("I25") or code.startswith("I50"):
            return "Circulatory"
        if code.startswith("J"):
            return "Respiratory"
        if code.startswith("N39") or code.startswith("N"):
            return "Genitourinary"
        if code.startswith("E87") or code.startswith("R55"):
            return "Other_Metabolic_Symptom"
        if code.startswith("Z"):
            return "Administrative_Followup"
        return "Other_ICD10"

    # ICD-9 style (numeric / numeric.numeric)
    try:
        base = float(code)
    except ValueError:
        return "Unknown"
    if 250.00 <= base < 251.00:
        return "Diabetes"
    if 390 <= base < 460 or base == 428.0:
        return "Circulatory"
    if 460 <= base < 520:
        return "Respiratory"
    if 580 <= base < 630 or base == 599.0:
        return "Genitourinary"
    if base == 780.2 or base == 276.1:
        return "Other_Metabolic_Symptom"
    if str(code).startswith("V"):
        return "Administrative_Followup"
    return "Other_ICD9"


def standardize_categorical(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"?": pd.NA, "": pd.NA, "None": pd.NA, "none": pd.NA})
    )


def clean_encounters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Remove duplicate records (exact duplicates from the raw extract)
    before = len(df)
    df = df.drop_duplicates(subset=["encounter_id"], keep="first")
    print(f"[CLEAN] removed {before - len(df):,} duplicate encounter rows")

    # 2. Standardize categoricals ('?' / blanks -> proper NA)
    cat_cols = ["race", "gender", "admission_type", "discharge_disposition",
                "admission_source", "medical_specialty", "max_glu_serum",
                "A1Cresult", "change", "diabetesMed", "readmitted", "age"]
    for c in cat_cols:
        df[c] = standardize_categorical(df[c])

    # 3. Parse & validate dates
    df["admit_datetime"] = pd.to_datetime(df["admit_datetime"], errors="coerce")
    df["discharge_datetime"] = pd.to_datetime(df["discharge_datetime"], errors="coerce")
    invalid_dates = df["discharge_datetime"] < df["admit_datetime"]
    print(f"[CLEAN] found {invalid_dates.sum():,} rows with discharge < admit; discharge set to admit+time_in_hospital")
    df.loc[invalid_dates, "discharge_datetime"] = (
        df.loc[invalid_dates, "admit_datetime"]
        + pd.to_timedelta(df.loc[invalid_dates, "time_in_hospital"].fillna(1), unit="D")
    )

    # 4. Handle missing numeric values (median imputation, tracked with flags)
    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                     "num_medications", "number_outpatient", "number_emergency",
                     "number_inpatient", "number_diagnoses"]
    for c in numeric_cols:
        missing_flag = df[c].isna()
        if missing_flag.any():
            df[f"{c}_was_missing"] = missing_flag.astype(int)
            df[c] = df[c].fillna(df[c].median())
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 5. Missing categorical -> explicit 'Unknown' bucket (informative missingness)
    for c in ["race", "gender", "age", "medical_specialty"]:
        n_missing = df[c].isna().sum()
        if n_missing:
            print(f"[CLEAN] {c}: imputing {n_missing:,} missing values as 'Unknown'")
        df[c] = df[c].fillna("Unknown")

    # 6. Standardize diagnosis codes -> unified ICD category taxonomy
    for c in ["diag_1", "diag_2", "diag_3"]:
        df[f"{c}_category"] = df[c].apply(categorize_icd)

    # 7. Convert dtypes explicitly
    df["time_in_hospital"] = df["time_in_hospital"].astype(int)
    for c in ["num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient",
              "number_diagnoses"]:
        df[c] = df[c].round().astype(int)

    return df


def clean_labs(labs: pd.DataFrame) -> pd.DataFrame:
    labs = labs.copy()
    before = len(labs)
    labs = labs.drop_duplicates(subset=["encounter_id"], keep="first")
    print(f"[CLEAN] labs: removed {before - len(labs):,} duplicate rows")

    # Outlier handling via IQR clipping on key clinical measures
    for c in ["hba1c_value", "ldl_cholesterol", "creatinine", "bmi", "systolic_bp", "diastolic_bp"]:
        q1, q3 = labs[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((labs[c] < lower) | (labs[c] > upper)).sum()
        if n_outliers:
            print(f"[CLEAN] labs.{c}: clipped {n_outliers:,} IQR outliers to [{lower:.1f}, {upper:.1f}]")
        labs[c] = labs[c].clip(lower, upper)
        labs[c] = labs[c].fillna(labs[c].median())

    labs["retrieved_at"] = pd.to_datetime(labs["retrieved_at"], errors="coerce")
    return labs


def join_sources(encounters: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    merged = encounters.merge(labs, on="encounter_id", how="left", indicator=True)
    match_rate = (merged["_merge"] == "both").mean()
    print(f"[TRANSFORM] joined encounters + labs on encounter_id | match rate: {match_rate:.1%}")
    merged["has_lab_data"] = (merged["_merge"] == "both").astype(int)
    merged = merged.drop(columns=["_merge"])

    # Impute unmatched lab values with global median (documented decision)
    for c in ["hba1c_value", "ldl_cholesterol", "creatinine", "bmi", "systolic_bp", "diastolic_bp"]:
        merged[c] = merged[c].fillna(merged[c].median())

    return merged


def run():
    encounters, labs = load_raw_to_sqlite_and_parquet()
    encounters_clean = clean_encounters(encounters)
    labs_clean = clean_labs(labs)
    merged = join_sources(encounters_clean, labs_clean)

    write_stage(encounters_clean, "interim", "encounters_clean")
    write_stage(labs_clean, "interim", "labs_clean")
    write_stage(merged, "interim", "merged_clean")

    print(f"\n[TRANSFORM] final cleaned/merged shape: {merged.shape}")
    print(f"[TRANSFORM] missing values remaining per column:\n{merged.isna().sum()[merged.isna().sum() > 0]}")
    return merged


if __name__ == "__main__":
    run()
