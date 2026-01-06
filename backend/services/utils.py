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
import math
import json
from fastapi.encoders import jsonable_encoder
import numpy as np

BASE_DATASET_DIR = "data/datasets"
MODELS_DIR = "models"
DATA_DIR = "data"

def datasetdir(dataset_id: str) -> str:
    path = os.path.join(BASE_DATASET_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

def raw_path(dataset_id: str) -> str:
    return os.path.join(DATA_DIR, f"{dataset_id}.csv")

def model_dir(dataset_id: str) -> str:
    path = os.path.join(MODELS_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_raw(dataset_id: str) -> pd.DataFrame:
    path = raw_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Raw dataset not found")
    return pd.read_csv(path)

def load_clean(dataset_id: str) -> pd.DataFrame:
    path = os.path.join(datasetdir(dataset_id), "clean.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "Clean dataset not found")
    return pd.read_csv(path)

def safe(obj):
    return jsonable_encoder(
        obj,
        custom_encoder={
            float: lambda x: None if (math.isnan(x) or math.isinf(x)) else x,
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
        },
    )
