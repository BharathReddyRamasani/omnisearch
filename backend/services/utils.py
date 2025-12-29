# backend/services/utils.py - CREATE/REPLACE THIS FILE
import os
import pandas as pd
from fastapi import HTTPException
import math

BASEDATADIR = "data/datasets"

def datasetdir(dataset_id: str) -> str:
    """Return dataset directory: data/datasets/{id}"""
    path = os.path.join(BASEDATADIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

def loaddf(dataset_id: str) -> pd.DataFrame:
    """Load raw dataset (checks both locations)."""
    # Priority: datasets folder first
    raw_path = os.path.join(datasetdir(dataset_id), "raw.csv")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    
    # Fallback: original upload location (your current app.py)
    fallback_path = f"data/{dataset_id}.csv"
    if os.path.exists(fallback_path):
        # Copy to standard location for consistency
        df = pd.read_csv(fallback_path)
        df.to_csv(raw_path, index=False)
        return df
    
    raise HTTPException(status_code=404, detail="Dataset not found")

def cleandataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning - strip strings, coerce numeric, drop all-null rows."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(how='all')

def safe_json(val):
    """Safe JSON serialization for NaN/inf."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val
