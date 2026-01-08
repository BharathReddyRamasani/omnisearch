# # import pandas as pd
# # import numpy as np
# # from backend.services.utils import load_raw

# # def generate_eda(dataset_id: str):
# #     df = load_raw(dataset_id)

# #     total_missing = int(df.isnull().sum().sum())
# #     missing_pct = (
# #         total_missing / (len(df) * len(df.columns)) * 100
# #         if len(df) > 0 else 0
# #     )

# #     quality_score = max(0, min(100, 100 - missing_pct))

# #     return {
# #         "status": "ok",
# #         "eda": {
# #             "rows": len(df),
# #             "columns": len(df.columns),
# #             "missing": df.isnull().sum().to_dict(),
# #             "missing_pct": round(missing_pct, 2),
# #             "quality_score": round(quality_score, 1),
# #             "summary": df.describe().round(2).replace({np.nan: None}).to_dict()
# #         }
# #     }


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN
# from backend.services.utils import load_raw, load_clean


# def _numeric_cols(df):
#     return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


# def detect_outliers_zscore(df, threshold=3.0):
#     out = {}
#     for col in _numeric_cols(df):
#         s = df[col].dropna()
#         if s.std() == 0:
#             continue
#         z = (s - s.mean()) / s.std()
#         idx = z[abs(z) > threshold].index.tolist()
#         out[col] = {
#             "count": len(idx),
#             "percentage": round(len(idx) / len(df) * 100, 2),
#         }
#     return out


# def detect_outliers_iqr(df):
#     out = {}
#     for col in _numeric_cols(df):
#         s = df[col].dropna()
#         q1, q3 = s.quantile(0.25), s.quantile(0.75)
#         iqr = q3 - q1
#         if iqr == 0:
#             continue
#         lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
#         idx = s[(s < lower) | (s > upper)].index.tolist()
#         out[col] = {
#             "count": len(idx),
#             "percentage": round(len(idx) / len(df) * 100, 2),
#             "lower": round(float(lower), 4),
#             "upper": round(float(upper), 4),
#         }
#     return out


# def detect_outliers_dbscan(df):
#     num_cols = _numeric_cols(df)
#     if len(num_cols) < 2:
#         return {}

#     X = df[num_cols].dropna()
#     if len(X) < 20:
#         return {}

#     X_scaled = StandardScaler().fit_transform(X)
#     labels = DBSCAN(eps=2.5, min_samples=5).fit_predict(X_scaled)

#     outliers = X.index[labels == -1].tolist()

#     return {
#         "method": "DBSCAN",
#         "outlier_count": len(outliers),
#         "percentage": round(len(outliers) / len(df) * 100, 2),
#         "features_used": num_cols,
#     }


# def safe_numeric_summary(s: pd.Series):
#     if s.std() == 0:
#         return {"mean": float(s.mean()), "std": 0.0, "min": float(s.min()), "max": float(s.max())}
#     return {
#         "mean": float(s.mean()),
#         "std": float(s.std()),
#         "min": float(s.min()),
#         "25%": float(s.quantile(0.25)),
#         "50%": float(s.quantile(0.50)),
#         "75%": float(s.quantile(0.75)),
#         "max": float(s.max()),
#     }


# def safe_categorical_summary(s: pd.Series):
#     if s.empty:
#         return {"unique": 0, "top": None}
#     unique_count = s.nunique()
#     return {
#         "unique": int(unique_count),
#         "top": s.mode().iloc[0] if unique_count > 0 and not s.mode().empty else None,
#     }


# def generate_eda(dataset_id: str):
#     # Prefer clean if available
#     try:
#         df = load_clean(dataset_id)
#         source = "clean"
#     except Exception:
#         df = load_raw(dataset_id)
#         source = "raw"

#     total_cells = df.size
#     total_missing = int(df.isna().sum().sum())
#     missing_pct = round((total_missing / max(1, total_cells)) * 100, 2)
#     dup_pct = df.duplicated().sum() / len(df) * 100
#     quality_score = round(max(0.0, 100 - missing_pct - (dup_pct / 5)), 1)  # Adjusted for duplicates (milder penalty)

#     # Compute summary on FULL data
#     summary = {}
#     for col in df.columns:
#         s = df[col]
#         if pd.api.types.is_numeric_dtype(s):
#             summary[col] = safe_numeric_summary(s.dropna())
#         else:
#             summary[col] = safe_categorical_summary(s)

#     # Sample for outliers
#     MAX_SAMPLE_ROWS = 30000
#     if len(df) > MAX_SAMPLE_ROWS:
#         df_sample = df.sample(MAX_SAMPLE_ROWS, random_state=42)
#     else:
#         df_sample = df.copy()

#     outliers = {
#         "zscore": detect_outliers_zscore(df_sample),
#         "iqr": detect_outliers_iqr(df_sample),
#         "cluster": detect_outliers_dbscan(df_sample),
#     }

#     return {
#         "status": "ok",
#         "eda": {
#             "rows": int(len(df)),
#             "columns": int(len(df.columns)),
#             "missing": df.isna().sum().to_dict(),
#             "missing_pct": missing_pct,
#             "quality_score": quality_score,
#             "summary": summary,
#             "outliers": outliers,
#             "etl_complete": source == "clean",
#         },
#     }
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from backend.services.utils import load_raw, load_clean


def _numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def detect_outliers_zscore(df, threshold=3.0):
    out = {}
    n = len(df)
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 10 or s.std() == 0:
            continue
        count = int(((s - s.mean()).abs() / s.std() > threshold).sum())
        out[col] = {"count": count, "percentage": round(count / n * 100, 2)}
    return out


def detect_outliers_iqr(df):
    out = {}
    n = len(df)
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lower) | (s > upper)).sum())
        out[col] = {
            "count": count,
            "percentage": round(count / n * 100, 2),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4),
        }
    return out


def detect_outliers_dbscan(df):
    cols = _numeric_cols(df)
    if len(cols) < 2:
        return {}

    X = df[cols].dropna()
    if len(X) < 50:
        return {}

    MAX_ROWS = 15000
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=42)

    labels = DBSCAN(eps=2.5, min_samples=10).fit_predict(
        StandardScaler().fit_transform(X)
    )
    count = int((labels == -1).sum())
    return {
        "method": "DBSCAN",
        "outlier_count": count,
        "percentage": round(count / len(df) * 100, 2),
        "sampled_rows": len(X),
    }


def generate_eda(dataset_id: str):
    try:
        df = load_clean(dataset_id)
        source = "clean"
    except Exception:
        df = load_raw(dataset_id)
        source = "raw"

    rows = len(df)
    total_cells = max(1, df.size)
    total_missing = int(df.isna().sum().sum())
    missing_pct = round(total_missing / total_cells * 100, 2)

    dup_pct = round(df.duplicated().sum() / max(1, rows) * 100, 2)
    quality_score = round(max(0.0, 100 - missing_pct - min(15.0, dup_pct)), 1)

    # Safe summary on full data
    summary = {}
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            if s.empty:
                summary[col] = None
            else:
                std = s.std()
                summary[col] = {
                    "mean": float(s.mean()),
                    "std": float(std) if std > 0 else 0.0,
                    "min": float(s.min()),
                    "25%": float(s.quantile(0.25)),
                    "50%": float(s.quantile(0.50)),
                    "75%": float(s.quantile(0.75)),
                    "max": float(s.max()),
                }
        else:
            summary[col] = {
                "unique": int(s.nunique()),
                "top": s.mode().iloc[0] if not s.mode().empty else None,
            }

    # Sample only for expensive computations
    MAX_SAMPLE = 30000
    df_sample = df.sample(min(MAX_SAMPLE, rows), random_state=42)

    return {
        "status": "ok",
        "eda": {
            "rows": rows,
            "columns": len(df.columns),
            "missing": df.isna().sum().to_dict(),
            "missing_pct": missing_pct,
            "duplicate_pct": dup_pct,
            "quality_score": quality_score,
            "summary": summary,
            "outliers": {
                "zscore": detect_outliers_zscore(df_sample),
                "iqr": detect_outliers_iqr(df_sample),
                "cluster": detect_outliers_dbscan(df_sample),
            },
            "etl_complete": source == "clean",
            "source": source,
        },
    }