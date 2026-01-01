# # backend/services/cleaning.py
# import os, json
# import pandas as pd
# import numpy as np
# from backend.services.utils import datasetdir, load_raw


# def full_etl(dataset_id: str):
#     df_raw = load_raw(dataset_id)
#     df_clean = df_raw.copy()

#     # ---------- RAW METRICS ----------
#     raw_metrics = {
#         "rows": len(df_raw),
#         "missing_total": int(df_raw.isnull().sum().sum()),
#         "duplicate_rows": int(df_raw.duplicated().sum())
#     }

#     # ---------- DEDUP ----------
#     df_clean = df_clean.drop_duplicates()

#     # ---------- MISSING ----------
#     missing_filled = 0
#     for col in df_clean.select_dtypes(include="number").columns:
#         before = df_clean[col].isnull().sum()
#         df_clean[col] = df_clean[col].fillna(df_clean[col].median())
#         missing_filled += before

#     for col in df_clean.select_dtypes(include="object").columns:
#         before = df_clean[col].isnull().sum()
#         if not df_clean[col].mode().empty:
#             df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
#         missing_filled += before

#     # ---------- OUTLIERS (IQR) ----------
#     outliers_fixed = 0
#     for col in df_clean.select_dtypes(include="number").columns:
#         q1 = df_clean[col].quantile(0.25)
#         q3 = df_clean[col].quantile(0.75)
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr

#         outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
#         outliers_fixed += int(outliers)

#         df_clean[col] = df_clean[col].clip(lower, upper)

#     # ---------- CLEAN METRICS ----------
#     clean_metrics = {
#         "rows": len(df_clean),
#         "missing_total": int(df_clean.isnull().sum().sum()),
#         "duplicate_rows": 0
#     }

#     # ---------- QUALITY SCORE ----------
#     raw_quality = max(0, 100 - (raw_metrics["missing_total"] / max(1, raw_metrics["rows"]) * 100))
#     clean_quality = max(0, 100 - (clean_metrics["missing_total"] / max(1, clean_metrics["rows"]) * 100))

#     # ---------- SAVE ----------
#     ddir = datasetdir(dataset_id)
#     df_clean.to_csv(os.path.join(ddir, "clean.csv"), index=False)

#     comparison = {
#         "raw_stats": raw_metrics,
#         "clean_stats": clean_metrics,
#         "improvements": {
#             "missing_values_filled": missing_filled,
#             "outliers_fixed": outliers_fixed,
#             "duplicates_removed": raw_metrics["duplicate_rows"]
#         },
#         "quality": {
#             "before": round(raw_quality, 2),
#             "after": round(clean_quality, 2)
#         },
#         "accuracy_lift_expected": round(min(20, (clean_quality - raw_quality)), 2)
#     }

#     with open(os.path.join(ddir, "comparison.json"), "w") as f:
#         json.dump(comparison, f, indent=2)

#     return {
#         "status": "success",
#         "comparison": comparison
#     }

import os, json
import pandas as pd
import numpy as np
from backend.services.utils import datasetdir, load_raw

def to_py(obj):
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
def full_etl(dataset_id: str):
    df_raw = load_raw(dataset_id)
    df_clean = df_raw.copy()

    raw_metrics = {
        "rows": int(len(df_raw)),
        "missing_total": int(df_raw.isna().sum().sum()),
        "duplicate_rows": int(df_raw.duplicated().sum()),
    }

    df_clean = df_clean.drop_duplicates()

    missing_filled = 0
    for col in df_clean.select_dtypes(include="number"):
        before = int(df_clean[col].isna().sum())
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        missing_filled += before

    for col in df_clean.select_dtypes(include="object"):
        before = int(df_clean[col].isna().sum())
        if not df_clean[col].mode().empty:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        missing_filled += before

    outliers_fixed = 0
    for col in df_clean.select_dtypes(include="number"):
        q1 = float(df_clean[col].quantile(0.25))
        q3 = float(df_clean[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(((df_clean[col] < lower) | (df_clean[col] > upper)).sum())
        outliers_fixed += count
        df_clean[col] = df_clean[col].clip(lower, upper)

    clean_metrics = {
        "rows": int(len(df_clean)),
        "missing_total": int(df_clean.isna().sum().sum()),
        "duplicate_rows": 0,
    }

    raw_quality = float(max(0, 100 - (raw_metrics["missing_total"] / max(1, raw_metrics["rows"]) * 100)))
    clean_quality = float(max(0, 100 - (clean_metrics["missing_total"] / max(1, clean_metrics["rows"]) * 100)))

    ddir = datasetdir(dataset_id)
    df_clean.to_csv(os.path.join(ddir, "clean.csv"), index=False)

    comparison = {
        "raw_stats": raw_metrics,
        "clean_stats": clean_metrics,
        "improvements": {
            "missing_values_filled": int(missing_filled),
            "outliers_fixed": int(outliers_fixed),
            "duplicates_removed": int(raw_metrics["duplicate_rows"]),
        },
        "quality": {
            "before": round(raw_quality, 2),
            "after": round(clean_quality, 2),
        },
        "accuracy_lift_expected": round(min(20.0, clean_quality - raw_quality), 2),
    }

    with open(os.path.join(ddir, "comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    return {
        "status": "ok",
        "comparison": comparison,
    }
