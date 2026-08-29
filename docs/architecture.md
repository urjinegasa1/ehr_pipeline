# Pipeline Architecture — Healthcare/EHR Readmission Risk Pipeline

## Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Source 1: CSV     │     │ Source 2: sim API  │     │                    │
│ ehr_encounters.csv│     │ ehr_lab_api_       │     │   INGESTION        │
│ (patient encounters)   │ extract.json (labs)│     │   (generate_raw_   │
└────────┬─────────┘     └─────────┬──────────┘     │   sources.py)      │
         │                         │                 └────────────────────┘
         ▼                         ▼
┌─────────────────────────────────────────────┐
│  STORAGE (storage.py)                        │
│  raw -> SQLite (warehouse.db) + Parquet      │
└───────────────────┬───────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  CLEANING / TRANSFORM (transform.py)         │
│  - dedupe, standardize categoricals          │
│  - ICD-9/10 -> unified diagnosis category    │
│  - IQR outlier clipping on labs              │
│  - join encounters + labs on encounter_id    │
└───────────────────┬───────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  FEATURE ENGINEERING (features.py)           │
│  16 engineered features + binary target      │
│  -> ML-ready dataset (SQLite+Parquet+CSV)    │
└───────────────────┬───────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  DATA QUALITY (quality.py, Great Expectations)│
│  18 expectations: ranges, nulls, categorical  │
│  domains, uniqueness -> fails DAG if unmet    │
└───────────────────┬───────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│  DISTRIBUTED PROCESSING (spark_processing.py)│
│  PySpark: window functions (patient sequence,│
│  days-since-prior-encounter), groupBy         │
│  aggregations, rapid-readmission cohort       │
└───────────────────┬───────────────────────────┘
                     ▼
              ML-ready dataset
      (data/processed/ehr_ml_ready.*)

Orchestrated end-to-end by Airflow DAG `ehr_healthcare_pipeline`
(dags/ehr_pipeline_dag.py), scheduled nightly at 02:00.
Every stage's output is versioned with DVC (raw + processed).
```

## Design decisions

1. **Two-source ingestion**: a primary CSV extract (mirrors the UCI Diabetes
   130-US Hospitals schema) plus a simulated hospital lab-results API feed,
   satisfying the "extract from ≥2 sources" requirement and matching the
   assignment's own "simulated healthcare API" guidance for this topic.
2. **Dual storage (SQLite + Parquet, plus CSV for the final output)**: SQLite
   gives ad-hoc SQL access for analysts; Parquet is used by Spark and Polars
   for fast columnar reads; CSV is kept for the final deliverable so it's
   directly usable by any ML tool without extra dependencies.
3. **ICD standardization**: rather than a full ICD-9↔ICD-10 GEM crosswalk
   (overkill for this dataset), diagnosis codes are mapped into six clinically
   meaningful categories (Diabetes, Circulatory, Respiratory, Genitourinary,
   Other_Metabolic_Symptom, Administrative_Followup), which is what the
   downstream model actually needs.
4. **Privacy-aware processing**: `patient_id` is retained only as a join key
   through the cleaning/feature stages and is dropped before the ML-ready
   output is written, per the assignment's "privacy considerations" /
   "anonymized" DVC-versioning requirement.
5. **Quality gate before Spark**: Great Expectations validation runs before
   the Spark aggregation step in the Airflow DAG. A failed suite raises and
   halts the DAG, so malformed data never reaches the ML-ready output —
   this ordering choice trades a bit of pipeline latency for correctness.
6. **Airflow task granularity**: one task per pipeline phase (not per
   dataset) keeps the DAG readable and lets retries be scoped narrowly —
   e.g. a transient lab-API failure only retries `extract`, not the whole
   pipeline.

## Reproducibility

See `README.md` for the exact commands (`python src/pipeline_main.py` for a
local end-to-end run, or the Airflow DAG for scheduled orchestration).
