"""
storage.py
----------
Phase: STORAGE

Loads the two raw sources (CSV + simulated API JSON) into:
  - SQLite (data/warehouse.db)   -> raw_encounters, raw_labs tables
  - Parquet (data/raw/*.parquet) -> columnar copy for fast Spark/Polars reads

Also exposes helpers for writing the cleaned / feature-engineered / ML-ready
outputs to both formats later in the pipeline (used by transform.py,
features.py, and pipeline_main.py).
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DB_PATH = BASE_DIR / "data" / "warehouse.db"

for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_conn():
    return sqlite3.connect(DB_PATH)


def load_raw_to_sqlite_and_parquet():
    # --- Source 1: CSV ---
    encounters = pd.read_csv(RAW_DIR / "ehr_encounters.csv")

    # --- Source 2: simulated API JSON ---
    with open(RAW_DIR / "ehr_lab_api_extract.json") as f:
        labs = pd.DataFrame(json.load(f))

    with get_conn() as conn:
        encounters.to_sql("raw_encounters", conn, if_exists="replace", index=False)
        labs.to_sql("raw_labs", conn, if_exists="replace", index=False)

    encounters.to_parquet(RAW_DIR / "raw_encounters.parquet", index=False)
    labs.to_parquet(RAW_DIR / "raw_labs.parquet", index=False)

    print(f"[STORAGE] raw_encounters: {len(encounters):,} rows -> SQLite + Parquet")
    print(f"[STORAGE] raw_labs:       {len(labs):,} rows -> SQLite + Parquet")
    return encounters, labs


def write_stage(df: pd.DataFrame, stage: str, name: str):
    """
    Writes a dataframe to both SQLite and Parquet for a given pipeline stage.
    stage: 'interim' | 'processed'
    """
    target_dir = INTERIM_DIR if stage == "interim" else PROCESSED_DIR
    parquet_path = target_dir / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False)

    with get_conn() as conn:
        table_name = f"{stage}_{name}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)

    print(f"[STORAGE] {stage}.{name}: {len(df):,} rows, {df.shape[1]} cols -> "
          f"SQLite(table={stage}_{name}) + {parquet_path.name}")
    return parquet_path


if __name__ == "__main__":
    load_raw_to_sqlite_and_parquet()
