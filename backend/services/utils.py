# # # backend/services/utils.py
# # import os
# # import pandas as pd
# # from fastapi import HTTPException

# # # Base directory for all datasets
# # BASE_DATASET_DIR = "data/datasets"


# # def datasetdir(dataset_id: str) -> str:
# #     """
# #     Returns the canonical dataset directory:
# #     data/datasets/<dataset_id>

# #     Creates it if it does not exist.
# #     """
# #     path = os.path.join(BASE_DATASET_DIR, dataset_id)
# #     os.makedirs(path, exist_ok=True)
# #     return path


# # def load_raw(dataset_id: str) -> pd.DataFrame:
# #     """
# #     Load raw uploaded CSV.
# #     """
# #     path = f"data/{dataset_id}.csv"
# #     if not os.path.exists(path):
# #         raise HTTPException(status_code=404, detail="Raw dataset not found")
# #     return pd.read_csv(path)


# # def load_clean(dataset_id: str) -> pd.DataFrame:
# #     """
# #     Load cleaned dataset.
# #     """
# #     path = os.path.join(datasetdir(dataset_id), "clean.csv")
# #     if not os.path.exists(path):
# #         raise HTTPException(status_code=404, detail="Clean dataset not found")
# #     return pd.read_csv(path)



# import os
# import math
# import json
# import pandas as pd
# import numpy as np
# from fastapi import HTTPException
# from fastapi.encoders import jsonable_encoder

# # =====================================================
# # BASE PATHS (ABSOLUTE, SAFE)
# # =====================================================
# BASE_DIR = os.getcwd()

# DATA_DIR = os.path.join(BASE_DIR, "data")
# DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
# MODELS_DIR = os.path.join(BASE_DIR, "models")

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(DATASETS_DIR, exist_ok=True)
# os.makedirs(MODELS_DIR, exist_ok=True)

# # =====================================================
# # PATH HELPERS
# # =====================================================
# def datasetdir(dataset_id: str) -> str:
#     """
#     Directory for cleaned datasets and ETL artifacts
#     """
#     path = os.path.join(DATASETS_DIR, dataset_id)
#     os.makedirs(path, exist_ok=True)
#     return path


# def raw_path(dataset_id: str) -> str:
#     """
#     Path to raw uploaded CSV
#     """
#     return os.path.join(DATA_DIR, f"{dataset_id}.csv")


# def model_dir(dataset_id: str) -> str:
#     """
#     Directory for trained models + metadata
#     """
#     path = os.path.join(MODELS_DIR, dataset_id)
#     os.makedirs(path, exist_ok=True)
#     return path

# # =====================================================
# # LOADERS
# # =====================================================
# def load_raw(dataset_id: str) -> pd.DataFrame:
#     """
#     Load raw dataset safely
#     """
#     path = raw_path(dataset_id)
#     if not os.path.exists(path):
#         raise HTTPException(404, "Raw dataset not found")

#     try:
#         return pd.read_csv(path)
#     except UnicodeDecodeError:
#         # fallback for real-world CSVs
#         return pd.read_csv(path, encoding="latin1")


# def load_clean(dataset_id: str) -> pd.DataFrame:
#     """
#     Load cleaned dataset
#     """
#     path = os.path.join(datasetdir(dataset_id), "clean.csv")
#     if not os.path.exists(path):
#         raise HTTPException(404, "Clean dataset not found")

#     return pd.read_csv(path)


# def load_meta(dataset_id: str) -> dict:
#     """
#     Load trained model metadata (REQUIRED for predict & EDA overlays)
#     """
#     meta_path = os.path.join(model_dir(dataset_id), "metadata.json")
#     if not os.path.exists(meta_path):
#         raise HTTPException(404, "Model metadata not found. Train model first.")

#     with open(meta_path, "r") as f:
#         return json.load(f)

# # =====================================================
# # SAFE JSON ENCODER (NO NAN / INF CRASHES)
# # =====================================================
# def safe(obj):
#     """
#     Ensures FastAPI responses never crash due to NaN / Inf / numpy types
#     """
#     return jsonable_encoder(
#         obj,
#         custom_encoder={
#             float: lambda x: None if (math.isnan(x) or math.isinf(x)) else x,
#             np.integer: int,
#             np.floating: lambda x: None if (math.isnan(x) or math.isinf(x)) else float(x),
#             np.bool_: bool,
#         },
#     )


import os
import math
import json
import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

# =====================================================
# BASE DIRECTORIES (CENTRALIZED)
# =====================================================
BASE_DATA_DIR = "data"
DATASETS_DIR = os.path.join(BASE_DATA_DIR, "datasets")
MODELS_DIR = "models"

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================================
# PATH HELPERS
# =====================================================
def datasetdir(dataset_id: str) -> str:
    """
    Directory holding cleaned dataset artifacts
    """
    path = os.path.join(DATASETS_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path


def raw_path(dataset_id: str) -> str:
    """
    Raw uploaded CSV location
    """
    return os.path.join(BASE_DATA_DIR, f"{dataset_id}.csv")


def model_dir(dataset_id: str) -> str:
    """
    Directory holding trained model + metadata
    """
    path = os.path.join(MODELS_DIR, dataset_id)
    os.makedirs(path, exist_ok=True)
    return path

# =====================================================
# LOADERS
# =====================================================
def load_raw(dataset_id: str) -> pd.DataFrame:
    """
    Load raw uploaded dataset
    """
    path = raw_path(dataset_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Raw dataset not found")
    return pd.read_csv(path)


def load_clean(dataset_id: str) -> pd.DataFrame:
    """
    Load cleaned dataset (after ETL)
    """
    path = os.path.join(datasetdir(dataset_id), "clean.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Clean dataset not found")
    return pd.read_csv(path)


def load_meta(dataset_id: str) -> dict:
    """
    Load trained model metadata
    """
    path = os.path.join(model_dir(dataset_id), "metadata.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model metadata not found")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =====================================================
# JSON SAFETY (CRITICAL)
# =====================================================
def safe(obj):
    """
    Safely encode any object for JSON response.
    Prevents:
    - NaN / Inf JSON crashes
    - NumPy type serialization errors
    """

    def _safe_float(x):
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return x

    return jsonable_encoder(
        obj,
        custom_encoder={
            float: _safe_float,
            np.integer: int,
            np.floating: float,
            np.bool_: bool,
        },
    )
