# Data Dictionary — EHR Readmission Risk Pipeline (Topic 4)

Final ML-ready table: `data/processed/ehr_ml_ready.{parquet,csv}`
(SQLite table: `processed_ehr_ml_ready`)

## Source columns (from raw encounters + labs, cleaned)

| Column | Type | Description |
|---|---|---|
| encounter_id | string | Primary key, one row per hospital encounter |
| race, gender | string | Patient demographics; `?`/blank standardized to `Unknown` |
| age | string | 10-year age bracket, e.g. `[50-60)` |
| admission_type | string | Emergency / Urgent / Elective / Newborn / Trauma Center / Not Available |
| discharge_disposition | string | Where the patient went after discharge |
| admission_source | string | Referral source for the admission |
| time_in_hospital | int | Length of stay in days |
| medical_specialty | string | Attending physician's specialty |
| num_lab_procedures, num_procedures, num_medications | int | Counts during the encounter |
| number_outpatient, number_emergency, number_inpatient | int | Utilization in the prior year |
| number_diagnoses | int | Number of diagnoses recorded |
| diag_1/2/3_category | string | ICD-9/10 codes standardized into: Diabetes, Circulatory, Respiratory, Genitourinary, Other_Metabolic_Symptom, Administrative_Followup, Unknown |
| max_glu_serum, A1Cresult | string | Lab test result buckets |
| change, diabetesMed | string | Whether diabetes medication was changed / is prescribed |
| hba1c_value, ldl_cholesterol, creatinine, bmi, systolic_bp, diastolic_bp | float | Lab/vitals from the secondary (simulated API) source, IQR-outlier-clipped |
| has_lab_data | int (0/1) | Whether the encounter matched a lab record on join |

## Engineered features (16, see `src/features.py`)

| Feature | Type | Rationale |
|---|---|---|
| length_of_stay_bucket | categorical | Non-linear LOS risk signal (short/medium/long/extended) |
| age_group_numeric | float | Ordinal age midpoint for models needing numeric age |
| comorbidity_index | int | Count of distinct diagnosis categories — proxy for patient complexity |
| diagnosis_complexity | float | Weighted blend of diagnosis count + procedures + comorbidities |
| polypharmacy_flag | int (0/1) | ≥10 meds — established readmission risk factor |
| prior_utilization_score | float | Weighted prior outpatient/ER/inpatient visits |
| is_emergency_admission | int (0/1) | Emergency admissions readmit at higher rates |
| had_recent_er_visit | int (0/1) | Any ER visit in the lookback window |
| abnormal_a1c_flag | int (0/1) | A1C result was >7 or >8 |
| abnormal_glucose_flag | int (0/1) | Glucose serum result was >200 or >300 |
| bmi_category | categorical | Clinical BMI banding |
| hypertension_flag | int (0/1) | Systolic ≥140 or diastolic ≥90 |
| renal_risk_flag | int (0/1) | Creatinine > 1.3 mg/dL |
| lab_completeness_score | float | Fraction of key lab fields present — data-quality-aware feature |
| readmission_risk_score | float | Composite weighted business risk score |
| diabetes_med_changed | int (0/1) | Medication changed AND patient is on diabetes meds |

## Target

| Column | Type | Description |
|---|---|---|
| readmitted_within_30d | int (0/1) | 1 if the patient was readmitted within 30 days (from raw `readmitted == '<30'`) |

## Columns dropped before the ML-ready output (privacy / leakage)

`patient_id` (pseudonymized identifier, not a feature), `readmitted` (raw leakage source of the target), `admit_datetime`/`discharge_datetime` (info captured in LOS features), `diag_1/2/3` (replaced by `_category` versions), `lab_source_system`/`retrieved_at` (source metadata).
