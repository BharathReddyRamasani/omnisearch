# # backend/services/utils.py
# import os
# import pandas as pd
# from fastapi import HTTPException

# # Base directory for all datasets
# BASE_DATASET_DIR = "data/datasets"


# def datasetdir(dataset_id: str) -> str:
#     """
#     Returns the canonical dataset directory:
#     data/datasets/<dataset_id>

#     Creates it if it does not exist.
#     """
#     path = os.path.join(BASE_DATASET_DIR, dataset_id)
#     os.makedirs(path, exist_ok=True)
#     return path


# def load_raw(dataset_id: str) -> pd.DataFrame:
#     """
#     Load raw uploaded CSV.
#     """
#     path = f"data/{dataset_id}.csv"
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Raw dataset not found")
#     return pd.read_csv(path)


# def load_clean(dataset_id: str) -> pd.DataFrame:
#     """
#     Load cleaned dataset.
#     """
#     path = os.path.join(datasetdir(dataset_id), "clean.csv")
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Clean dataset not found")
#     return pd.read_csv(path)

import os
import pandas as pd
from fastapi import HTTPException

BASE_DATASET_DIR = "data/datasets"

def datasetdir(dataset_id: str) -> str:
    path = os.path.join(BASE_DATASET_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_raw(dataset_id: str) -> pd.DataFrame:
    path = f"data/{dataset_id}.csv"
    if not os.path.exists(path):
        raise HTTPException(404, "Raw dataset not found")
    return pd.read_csv(path)

def load_clean(dataset_id: str) -> pd.DataFrame:
    path = os.path.join(datasetdir(dataset_id), "clean.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "Clean dataset not found")
    return pd.read_csv(path)
