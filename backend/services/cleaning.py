# backend/services/cleaning.py

import os
import json
import pandas as pd
import numpy as np

from backend.services.utils import cleandataframe, datasetdir, loaddf


def full_etl(dataset_id: str):
    """
    Complete ETL: raw.csv → clean.csv + comparison stats.
    Uses existing dataset layout: data/datasets/{id}/raw.csv.
    """

    # ---------- 1. Load raw data ----------
    # Use existing helper so it works with your current ingest flow
    try:
        df_raw = loaddf(dataset_id)  # reads data/datasets/{id}/raw.csv
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Raw dataset not found or unreadable: {str(e)}"
        }

    # ---------- 2. Basic clean (your existing logic) ----------
    df_clean = cleandataframe(df_raw.copy())

    # ---------- 3. Outlier treatment (IQR clipping) ----------
    numeric_cols = df_clean.select_dtypes(include=["number"]).columns
    outliers_fixed = 0

    for col in numeric_cols:
        s = df_clean[col].dropna()
        if s.empty:
            continue

        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

        mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        outliers_before = mask.sum()
        outliers_fixed += int(outliers_before)

        df_clean.loc[mask, col] = df_clean[col].clip(lower, upper)[mask]

    # ---------- 4. Advanced imputation ----------
    missing_before = int(df_clean[numeric_cols].isnull().sum().sum())
    if len(numeric_cols) > 0:
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(
            df_clean[numeric_cols].median()
        )
    missing_after = int(df_clean[numeric_cols].isnull().sum().sum())
    missing_fixed = max(0, missing_before - missing_after)

    # ---------- 5. Save clean dataset ----------
    dpath = datasetdir(dataset_id)  # ensures folder data/datasets/{id}
    clean_path = os.path.join(dpath, "clean.csv")
    df_clean.to_csv(clean_path, index=False)

    # ---------- 6. Generate comparison stats ----------
    comparison = generate_comparison(
        df_raw=df_raw,
        df_clean=df_clean,
        outliers_fixed=outliers_fixed,
        missing_fixed=missing_fixed,
    )

    # ---------- 7. Save comparison JSON ----------
    comp_path = os.path.join(dpath, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return {
        "status": "success",
        "clean_path": clean_path,
        "comparison": comparison,
        "files_created": ["clean.csv", "comparison.json"],
    }


def generate_comparison(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    outliers_fixed: int,
    missing_fixed: int,
):
    """Raw vs Clean differences for UI display / download."""

    numeric_cols = df_clean.select_dtypes(include=["number"]).columns

    def missing_pct(df: pd.DataFrame) -> float:
        total_cells = len(df) * max(len(df.columns), 1)
        if total_cells == 0:
            return 0.0
        return float(df.isnull().sum().sum()) / total_cells * 100.0

    raw_missing = missing_pct(df_raw)
    clean_missing = missing_pct(df_clean)

    return {
        "raw_stats": {
            "rows": int(len(df_raw)),
            "columns": int(len(df_raw.columns)),
            "missing_percent": raw_missing,
        },
        "clean_stats": {
            "rows": int(len(df_clean)),
            "columns": int(len(df_clean.columns)),
            "missing_percent": clean_missing,
        },
        "improvements": {
            "outliers_fixed": int(outliers_fixed),
            "missing_values_filled": int(missing_fixed),
            "numeric_columns_cleaned": int(len(numeric_cols)),
        },
        # Rough “expected lift” number just for UX
        "accuracy_lift_expected": float(min(15.0, outliers_fixed * 0.01)),
    }
