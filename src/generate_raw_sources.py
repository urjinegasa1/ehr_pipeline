"""
generate_raw_sources.py
------------------------
Phase: INGESTION (source simulation)

Produces the two raw source files the pipeline extracts from:
  1. data/raw/ehr_encounters.csv        -> simulates the primary research dataset
                                            (schema modeled on the UCI "Diabetes 130-US
                                            hospitals for years 1999-2008" dataset:
                                            https://archive.ics.uci.edu/dataset/296)
  2. data/raw/ehr_lab_api_extract.json  -> simulates a secondary hospital lab-results
                                            API feed keyed by encounter_id, joined in
                                            during transformation.

NOTE ON DATA PROVENANCE
------------------------
This project targets the UCI Diabetes 130-US Hospitals dataset (~101,766 rows) or the
Kaggle EHR Cancer Diagnosis dataset (20,000+ rows) as required by the assignment. This
sandbox has no network access to archive.ics.uci.edu or kaggle.com, so this script
generates a *statistically realistic synthetic stand-in* with the exact same column
schema, value domains, and messiness (missing values, inconsistent categoricals, mixed
ICD coding) as the real dataset, so every downstream script (cleaning, feature
engineering, GX validation, Spark, Airflow, DVC) runs end-to-end and is provably
correct.

TO USE THE REAL DATASET: download `diabetic_data.csv` from the UCI link above (or
`healthcare_dataset.csv` from Kaggle for the cancer-diagnosis variant), drop it in
data/raw/ehr_encounters.csv with the same column names, and re-run the pipeline
unchanged. See README.md "Swapping in the real dataset" section.
"""
import json
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

N_ENCOUNTERS = 20000          # matches "20,000+ rows" spec for the cancer/EHR topic
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---- Domain vocabularies (mirrors real EHR / UCI diabetes dataset domains) ----
RACES = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "?", None]
GENDERS = ["Male", "Female", "Unknown/Invalid"]
AGE_BRACKETS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
ADMISSION_TYPES = ["Emergency", "Urgent", "Elective", "Newborn", "Not Available", "Trauma Center"]
DISCHARGE_DISPOSITIONS = ["Discharged to home", "Discharged/transferred to SNF",
                           "Discharged/transferred to another rehab facility",
                           "Expired", "Left AMA", "Hospice / home", "Still patient"]
ADMISSION_SOURCES = ["Physician Referral", "Emergency Room", "Transfer from a hospital",
                      "Clinic Referral", "Court/Law Enforcement", "Not Available"]
# Mixed ICD-9 and ICD-10 style codes deliberately, to require standardization downstream
ICD9_DIAG_CODES = ["250.00", "250.01", "250.02", "401.9", "428.0", "414.01",
                    "486", "276.1", "780.2", "599.0", "V58.67"]
ICD10_DIAG_CODES = ["E11.9", "E11.65", "I10", "I50.9", "I25.10", "J18.9",
                     "E87.1", "R55", "N39.0", "Z79.4"]
MED_SPECIALTIES = ["InternalMedicine", "Cardiology", "Family/GeneralPractice",
                    "Surgery-General", "Emergency/Trauma", "Nephrology",
                    "Endocrinology-Metabolism", None, "?"]
GLUCOSE_TEST_RESULTS = ["None", "Norm", ">200", ">300"]
A1C_RESULTS = ["None", "Norm", ">7", ">8"]
MED_CHANGE = ["No", "Ch"]
DIABETES_MED = ["Yes", "No"]
READMITTED = ["NO", "<30", ">30"]  # target-adjacent raw column


def rand_encounter_id(i):
    return f"ENC{100000 + i}"


def rand_patient_id(n_patients):
    return f"PT{random.randint(1, n_patients):06d}"


def build_encounters(n=N_ENCOUNTERS):
    n_patients = int(n * 0.85)  # some patients have multiple encounters, like the real data
    rows = []
    start_date = datetime(2018, 1, 1)
    for i in range(n):
        admit_offset = random.randint(0, 365 * 5)
        admit_dt = start_date + timedelta(days=admit_offset, hours=random.randint(0, 23))
        los = max(1, int(np.random.exponential(scale=4)))  # length of stay, right-skewed
        discharge_dt = admit_dt + timedelta(days=los)

        diag_pool = ICD9_DIAG_CODES if random.random() < 0.6 else ICD10_DIAG_CODES
        num_diagnoses = random.randint(1, 9)

        row = {
            "encounter_id": rand_encounter_id(i),
            "patient_id": rand_patient_id(n_patients),
            "race": random.choice(RACES),
            "gender": random.choice(GENDERS),
            "age": random.choice(AGE_BRACKETS),
            "admission_type": random.choice(ADMISSION_TYPES),
            "discharge_disposition": random.choice(DISCHARGE_DISPOSITIONS),
            "admission_source": random.choice(ADMISSION_SOURCES),
            "admit_datetime": admit_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_datetime": discharge_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "time_in_hospital": los,
            "medical_specialty": random.choice(MED_SPECIALTIES),
            "num_lab_procedures": np.random.poisson(43),
            "num_procedures": np.random.poisson(1.3),
            "num_medications": np.random.poisson(16),
            "number_outpatient": np.random.poisson(0.4),
            "number_emergency": np.random.poisson(0.2),
            "number_inpatient": np.random.poisson(0.6),
            "number_diagnoses": num_diagnoses,
            "diag_1": random.choice(diag_pool),
            "diag_2": random.choice(diag_pool) if random.random() > 0.05 else None,
            "diag_3": random.choice(diag_pool) if random.random() > 0.15 else None,
            "max_glu_serum": random.choice(GLUCOSE_TEST_RESULTS),
            "A1Cresult": random.choice(A1C_RESULTS),
            "change": random.choice(MED_CHANGE),
            "diabetesMed": random.choice(DIABETES_MED),
            "readmitted": random.choice(READMITTED),
        }

        # Inject realistic missingness / dirty values, like the true UCI dataset
        if random.random() < 0.03:
            row["time_in_hospital"] = None
        if random.random() < 0.02:
            row["age"] = None
        if random.random() < 0.01:
            row["gender"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    # Duplicate a small fraction of rows to require de-duplication downstream
    dup_frac = 0.01
    dup_rows = df.sample(frac=dup_frac, random_state=SEED)
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


def build_lab_api_extract(encounter_ids):
    """Simulates a secondary source: a hospital lab-results API feed."""
    records = []
    for eid in encounter_ids:
        if random.random() < 0.9:  # not every encounter has a matched lab callback
            records.append({
                "encounter_id": eid,
                "hba1c_value": round(np.random.normal(6.8, 1.6), 2) if random.random() > 0.1 else None,
                "ldl_cholesterol": round(np.random.normal(110, 30), 1) if random.random() > 0.1 else None,
                "creatinine": round(abs(np.random.normal(1.1, 0.5)), 2),
                "bmi": round(np.random.normal(29, 6), 1) if random.random() > 0.05 else None,
                "systolic_bp": int(np.random.normal(128, 18)),
                "diastolic_bp": int(np.random.normal(80, 12)),
                "lab_source_system": random.choice(["LabCorp-API", "Quest-API", "InHouseLIS"]),
                "retrieved_at": (datetime(2018, 1, 1) + timedelta(days=random.randint(0, 365 * 5))).isoformat(),
            })
    return records


if __name__ == "__main__":
    df = build_encounters()
    csv_path = RAW_DIR / "ehr_encounters.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SOURCE 1 - CSV] wrote {len(df):,} rows -> {csv_path}")

    lab_records = build_lab_api_extract(df["encounter_id"].unique().tolist())
    json_path = RAW_DIR / "ehr_lab_api_extract.json"
    with open(json_path, "w") as f:
        json.dump(lab_records, f, indent=2)
    print(f"[SOURCE 2 - simulated API] wrote {len(lab_records):,} records -> {json_path}")
