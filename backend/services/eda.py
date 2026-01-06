import pandas as pd
import numpy as np
from backend.services.utils import load_raw

def generate_eda(dataset_id: str):
    df = load_raw(dataset_id)

    total_missing = int(df.isnull().sum().sum())
    missing_pct = (
        total_missing / (len(df) * len(df.columns)) * 100
        if len(df) > 0 else 0
    )

    quality_score = max(0, min(100, 100 - missing_pct))

    return {
        "status": "ok",
        "eda": {
            "rows": len(df),
            "columns": len(df.columns),
            "missing": df.isnull().sum().to_dict(),
            "missing_pct": round(missing_pct, 2),
            "quality_score": round(quality_score, 1),
            "summary": df.describe().round(2).replace({np.nan: None}).to_dict()
        }
    }
