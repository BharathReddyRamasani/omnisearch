# import pandas as pd
# import numpy as np
# from backend.services.utils import load_raw

# def generate_eda(dataset_id: str):
#     df = load_raw(dataset_id)

#     total_missing = int(df.isnull().sum().sum())
#     missing_pct = (
#         total_missing / (len(df) * len(df.columns)) * 100
#         if len(df) > 0 else 0
#     )

#     quality_score = max(0, min(100, 100 - missing_pct))

#     return {
#         "status": "ok",
#         "eda": {
#             "rows": len(df),
#             "columns": len(df.columns),
#             "missing": df.isnull().sum().to_dict(),
#             "missing_pct": round(missing_pct, 2),
#             "quality_score": round(quality_score, 1),
#             "summary": df.describe().round(2).replace({np.nan: None}).to_dict()
#         }
#     }



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from backend.services.utils import load_raw


def _numeric_cols(df):
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def detect_outliers_zscore(df, threshold=3.0):
    out = {}
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if s.std() == 0:
            continue
        z = (s - s.mean()) / s.std()
        idx = z[abs(z) > threshold].index.tolist()
        out[col] = {
            "count": len(idx),
            "percentage": round(len(idx) / len(df) * 100, 2),
        }
    return out


def detect_outliers_iqr(df):
    out = {}
    for col in _numeric_cols(df):
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        idx = s[(s < lower) | (s > upper)].index.tolist()
        out[col] = {
            "count": len(idx),
            "percentage": round(len(idx) / len(df) * 100, 2),
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4),
        }
    return out


def detect_outliers_dbscan(df):
    num_cols = _numeric_cols(df)
    if len(num_cols) < 2:
        return {}

    X = df[num_cols].dropna()
    if len(X) < 20:
        return {}

    X_scaled = StandardScaler().fit_transform(X)
    labels = DBSCAN(eps=2.5, min_samples=5).fit_predict(X_scaled)

    outliers = X.index[labels == -1].tolist()

    return {
        "method": "DBSCAN",
        "outlier_count": len(outliers),
        "percentage": round(len(outliers) / len(df) * 100, 2),
        "features_used": num_cols,
    }


def generate_eda(dataset_id: str):
    df = load_raw(dataset_id)

    total_missing = int(df.isna().sum().sum())
    missing_pct = (total_missing / max(1, df.size)) * 100
    quality_score = max(0.0, 100 - missing_pct)

    return {
        "status": "ok",
        "eda": {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "missing": df.isna().sum().to_dict(),
            "missing_pct": round(missing_pct, 2),
            "quality_score": round(quality_score, 1),
            "summary": (
                df.describe(include="all")
                .replace({np.nan: None, np.inf: None, -np.inf: None})
                .to_dict()
            ),
            "outliers": {
                "zscore": detect_outliers_zscore(df),
                "iqr": detect_outliers_iqr(df),
                "cluster": detect_outliers_dbscan(df),
            },
        },
    }
