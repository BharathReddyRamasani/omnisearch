from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import json
import math
import base64
from io import BytesIO

import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# =====================================================
# APP SETUP
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DATA_DIR = "data/datasets"
os.makedirs(BASE_DATA_DIR, exist_ok=True)


# =====================================================
# UTILS (DO NOT TOUCH)
# =====================================================
def dataset_dir(dataset_id: str) -> str:
    return os.path.join(BASE_DATA_DIR, dataset_id)


def load_df(dataset_id: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir(dataset_id), "raw.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

        # try numeric conversion safely
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def safe_json(val):
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


# =====================================================
# UPLOAD (DAY 1)
# =====================================================
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    dataset_id = file.filename.replace(".", "_")
    dpath = dataset_dir(dataset_id)
    os.makedirs(dpath, exist_ok=True)

    try:
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.file)
        else:
            df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    df = clean_dataframe(df)
    df.to_csv(os.path.join(dpath, "raw.csv"), index=False)

    schema = {c: str(df[c].dtype) for c in df.columns}
    with open(os.path.join(dpath, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }


# =====================================================
# SCHEMA
# =====================================================
@app.get("/schema")
def get_schema(dataset_id: str):
    path = os.path.join(dataset_dir(dataset_id), "schema.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Schema not found")

    with open(path) as f:
        schema = json.load(f)

    return {"status": "ok", "schema": schema}


# =====================================================
# EDA (BEFORE + AFTER, OUTLIERS, PLOTS)
# =====================================================
@app.get("/eda")
def run_eda(dataset_id: str):
    df_raw = load_df(dataset_id)
    df = clean_dataframe(df_raw)

    missing = df.isnull().sum().to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    num_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    before_summary = {}
    after_summary = {}
    outliers_info = {}
    before_plots = {}
    after_plots = {}

    for col in num_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        # BEFORE
        before_summary[col] = series.describe().to_dict()

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = series[(series < lower) | (series > upper)]

        outliers_info[col] = {
            "count": int(outliers.count()),
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }

        fig, axes = plt.subplots(1, 2, figsize=(8, 3))
        series.hist(ax=axes[0], bins=20)
        axes[0].set_title(f"{col} (Before)")
        axes[1].boxplot(series, vert=False)
        axes[1].set_title("Outliers")

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        before_plots[col] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        # AFTER
        clipped = series.clip(lower, upper)
        after_summary[col] = clipped.describe().to_dict()

        fig, ax = plt.subplots(figsize=(4, 3))
        clipped.hist(ax=ax, bins=20)
        ax.set_title(f"{col} (After)")

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        after_plots[col] = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return {
        "status": "ok",
        "eda": {
            "missing": missing,
            "dtypes": dtypes,
            "before": {
                "summary": before_summary,
                "outliers": outliers_info,
                "plots": before_plots
            },
            "after": {
                "summary": after_summary,
                "plots": after_plots
            }
        }
    }

# =====================================================
# TRAIN (DAY 5)
# =====================================================
@app.post("/train")
def train_model(dataset_id: str, target: str):
    df = clean_dataframe(load_df(dataset_id))

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid target column")

    # TARGET MUST NOT HAVE NaN
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    for col in cat_cols:
        X[col] = X[col].astype(str)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    if y.dtype == "object" or y.nunique() <= 10:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        task = "classification"
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        task = "regression"

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    try:
        pipeline.fit(X, y)
    except Exception as e:
        return {
            "status": "failed",
            "stage": "training",
            "error": str(e)
        }

    dpath = dataset_dir(dataset_id)
    joblib.dump(pipeline, os.path.join(dpath, "model.pkl"))

    meta = {
        "target": target,
        "task": task,
        "features": list(X.columns)
    }

    with open(os.path.join(dpath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "ok",
        "task": task,
        "model_saved": True
    }


# =====================================================
# PREDICT (DAY 6)
# =====================================================
@app.post("/predict")
def predict(dataset_id: str, input_data: dict):
    model_path = os.path.join(dataset_dir(dataset_id), "model.pkl")

    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model not trained")

    pipeline = joblib.load(model_path)

    df = pd.DataFrame([input_data])
    df = clean_dataframe(df)

    pred = pipeline.predict(df)

    return {
        "status": "ok",
        "prediction": safe_json(pred[0])
    }
