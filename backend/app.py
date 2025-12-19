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
        df[col] = pd.to_numeric(df[col], errors="ignore")
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
# META (DAY-7)
# =====================================================
@app.get("/meta")
def get_meta(dataset_id: str):
    path = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Train the model first")

    with open(path) as f:
        return json.load(f)

# =====================================================
# EDA (DAY-2/3/4)
# =====================================================
@app.get("/eda")
def run_eda(dataset_id: str):
    df = clean_dataframe(load_df(dataset_id))

    missing = df.isnull().sum().to_dict()
    dtypes = {c: str(df[c].dtype) for c in df.columns}

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    before_summary, after_summary = {}, {}
    outliers, plots_before, plots_after = {}, {}, {}

    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        before_summary[col] = s.describe().to_dict()

        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

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
            "missing": missing,
            "dtypes": dtypes,
            "before": {"summary": before_summary, "outliers": outliers, "plots": plots_before},
            "after": {"summary": after_summary, "plots": plots_after}
        }
    }

# =====================================================
# TRAIN (DAY-5)
# =====================================================
@app.post("/train")
def train_model(dataset_id: str, target: str):
    df = clean_dataframe(load_df(dataset_id))

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Invalid target")

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    X = df.drop(columns=[target])
    y = df[target]

    # ---------------- DROP HIGH-NULL / USELESS COLUMNS ----------------
    DROP_COLS = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "GarageYrBlt"]
    X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

    # ---------------- SPLIT NUM / CAT ----------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Limit high-cardinality categorical columns
    cat_cols = [c for c in cat_cols if X[c].nunique() <= 20]

    for c in cat_cols:
        X[c] = X[c].astype(str)

    # ---------------- PREPROCESSOR ----------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    # ---------------- MODEL SELECTION ----------------
    if y.dtype == "object" or y.nunique() <= 10:
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )
        task = "classification"
    else:
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )
        task = "regression"

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    # ---------------- TRAIN ----------------
    try:
        pipeline.fit(X, y)
    except Exception as e:
        return {
            "status": "failed",
            "stage": "training",
            "error": str(e)
        }

    # ---------------- FEATURE IMPORTANCE (DAY-7 CORE) ----------------
    model_step = pipeline.named_steps["model"]
    top_features = list(X.columns)[:6]  # fallback

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
        prep = pipeline.named_steps["prep"]

        feature_names = []

        # numeric features
        if "num" in prep.named_transformers_:
            feature_names.extend(num_cols)

        # categorical (one-hot expanded)
        if "cat" in prep.named_transformers_:
            enc = prep.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = enc.get_feature_names_out(cat_cols).tolist()
            feature_names.extend(cat_feature_names)

        # sort by importance
        fi = sorted(
            zip(feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        )

        # map back to ORIGINAL columns
        top_original_features = []
        for fname, _ in fi:
            base = fname.split("_")[0]
            if base in X.columns and base not in top_original_features:
                top_original_features.append(base)

        if top_original_features:
            top_features = top_original_features[:6]

    # ---------------- SAVE ----------------
    dpath = dataset_dir(dataset_id)
    joblib.dump(pipeline, os.path.join(dpath, "model.pkl"))

    meta = {
        "target": target,
        "task": task,
        "features": list(X.columns),
        "top_features": top_features
    }

    with open(os.path.join(dpath, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "status": "ok",
        "task": task,
        "top_features": top_features
    }


# =====================================================
# PREDICT (DAY-6)
# =====================================================
@app.post("/predict")
def predict(dataset_id: str, input_data: dict):
    model_path = os.path.join(dataset_dir(dataset_id), "model.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Model not trained")

    pipeline = joblib.load(model_path)
    df = clean_dataframe(pd.DataFrame([input_data]))
    pred = pipeline.predict(df)

    return {"status": "ok", "prediction": safe_json(pred[0])}
