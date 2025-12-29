# utils.py - NEW FILE (backend/utils.py)
import os
import pandas as pd

def datasetdir(dataset_id):
    """Return dataset directory path"""
    return f"data/datasets/{dataset_id}"

def loaddf(dataset_id):
    """Load dataset - raw or clean"""
    raw_path = f"data/datasets/{dataset_id}/raw.csv"
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return pd.read_csv(f"data/{dataset_id}.csv")  # Fallback

def cleandataframe(df):
    """Basic cleaning - your existing function"""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(how='all')
