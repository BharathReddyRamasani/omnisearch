import pandas as pd
import os

BASE_DATA_DIR = "data/datasets"

def load_df(dataset_id):
    path = os.path.join(BASE_DATA_DIR, dataset_id, "raw.csv")
    return pd.read_csv(path)

def get_columns(dataset_id):
    df = load_df(dataset_id)
    return df.columns.tolist()

def get_schema(dataset_id):
    df = load_df(dataset_id)

    features = []
    for col, dtype in df.dtypes.items():
        features.append({
            "name": col,
            "dtype": str(dtype)
        })

    return {
        "features": features
    }
