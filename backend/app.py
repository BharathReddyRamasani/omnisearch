from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os, json, math, base64
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)

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
# UTILS
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
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    return df

def safe_json(val):
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val

# =====================================================
# UPLOAD (DAY-1)
# =====================================================
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    dataset_id = file.filename.replace(".", "_")
    dpath = dataset_dir(dataset_id)
    os.makedirs(dpath, exist_ok=True)

    if file.filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file.file)
    else:
        df = pd.read_csv(file.file)

    df = clean_dataframe(df)
    df.to_csv(os.path.join(dpath, "raw.csv"), index=False)

    schema = {c: str(df[c].dtype) for c in df.columns}
    json.dump(schema, open(os.path.join(dpath, "schema.json"), "w"), indent=2)

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
    return {"status": "ok", "schema": json.load(open(path))}

# =====================================================
# META
# =====================================================
@app.get("/meta")
def get_meta(dataset_id: str):
    path = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Train the model first")
    return json.load(open(path))

# =====================================================
# EDA (DAY-2/3/4)
# =====================================================
@app.get("/eda")
def run_eda(dataset_id: str):
    df = clean_dataframe(load_df(dataset_id))

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    before_summary, after_summary = {}, {}
    plots_before, plots_after, outliers = {}, {}, {}

    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        before_summary[col] = s.describe().to_dict()

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers[col] = {
            "count": int(((s < low) | (s > high)).sum()),
            "lower": float(low),
            "upper": float(high)
        }

        fig, ax = plt.subplots()
        s.hist(ax=ax)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plots_before[col] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        clipped = s.clip(low, high)
        after_summary[col] = clipped.describe().to_dict()

        fig, ax = plt.subplots()
        clipped.hist(ax=ax)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        plots_after[col] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {
        "status": "ok",
        "eda": {
            "missing": df.isnull().sum().to_dict(),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "before": {
                "summary": before_summary,
                "outliers": outliers,
                "plots": plots_before
            },
            "after": {
                "summary": after_summary,
                "plots": plots_after
            }
        }
    }

# =====================================================
# TRAIN + EVALUATE (DAY-5 → DAY-9)
# =====================================================
@app.post("/train")
def train_model(dataset_id: str, target: str):
    df = clean_dataframe(load_df(dataset_id))

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid target")

    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]

    DROP_COLS = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "GarageYrBlt"]
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if X[c].nunique() <= 20]

    for c in cat_cols:
        X[c] = X[c].astype(str)

    features = num_cols + cat_cols

    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ])

    # ---------------- TASK TYPE ----------------
    if y.dtype == "object" or y.nunique() <= 10:
        task = "classification"
        candidates = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    else:
        task = "regression"
        candidates = [
            ("LinearRegression", LinearRegression()),
            ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42))
        ]

    X_train, X_test, y_train, y_test = train_test_split(
        X[features], y, test_size=0.2, random_state=42
    )

    best_model, best_score, best_name = None, -1e9, None

    for name, model in candidates:
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        score = accuracy_score(y_test, preds) if task == "classification" else r2_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = pipe
            best_name = name

    preds = best_model.predict(X_test)

    # ---------------- METRICS ----------------
    if task == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist()
        }
    else:
        metrics = {
            "rmse": mean_squared_error(y_test, preds, squared=False),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }

    # ---------------- FEATURE IMPORTANCE ----------------
    top_features = features[:6]
    model_step = best_model.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        names = num_cols[:]

        if cat_cols:
            enc = best_model.named_steps["prep"].named_transformers_["cat"].named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cat_cols))

        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        mapped = []
        for fname, _ in ranked:
            base = fname.split("_")[0]
            if base in features and base not in mapped:
                mapped.append(base)
        if mapped:
            top_features = mapped[:6]

    # ---------------- MODEL VERSIONING ----------------
    dpath = dataset_dir(dataset_id)
    meta_path = os.path.join(dpath, "meta.json")

    version = 1
    if os.path.exists(meta_path):
        old_meta = json.load(open(meta_path))
        version = old_meta.get("current_version", 1) + 1

    model_name = f"model_v{version}.pkl"
    joblib.dump(best_model, os.path.join(dpath, model_name))

    meta = {
        "target": target,
        "task": task,
        "features": features,
        "top_features": top_features,
        "best_model": best_name,
        "metrics": metrics,
        "current_version": version,
        "model_file": model_name
    }

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "ok",
        "task": task,
        "best_model": best_name,
        "current_version": version,
        "metrics": metrics
    }

# =====================================================
# PREDICT (DAY-6 / DAY-7)
# =====================================================
@app.post("/predict")
def predict(dataset_id: str, input_data: dict):
    meta = json.load(open(os.path.join(dataset_dir(dataset_id), "meta.json")))
    model = joblib.load(os.path.join(dataset_dir(dataset_id), "model.pkl"))

    for col in meta["features"]:
        input_data.setdefault(col, None)

    df = clean_dataframe(pd.DataFrame([input_data]))
    pred = model.predict(df)

    return {"status": "ok", "prediction": safe_json(pred[0])}
