import os
import pandas as pd
import math
import json
from fastapi import HTTPException

BASE_DATA_DIR = "data/datasets"

def dataset_dir(dataset_id: str) -> str:
    path = os.path.join(BASE_DATA_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_df(dataset_id: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir(dataset_id), "raw.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return pd.read_csv(path)

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    df = df.drop_duplicates()
    return df

def safe_json(val):
    """JSON safe converter"""
    import numpy as np
    if isinstance(val, float):
        if np.isnan(val) or np.isinf(val):
            return None
    elif isinstance(val, np.integer):
        return int(val)
    elif isinstance(val, np.floating):
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    return val


def validate_target(series: pd.Series) -> str:
    n = len(series)
    if series.isnull().mean() > 0.3:
        return "Target has too many missing values (>30%)"
    if series.nunique() / n > 0.9:
        return "Target is almost unique (ID-like column)"
    if series.nunique() < 2:
        return "Target has no meaningful variation"
    return None
