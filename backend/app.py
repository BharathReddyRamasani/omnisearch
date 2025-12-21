# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware

# import os, json, math, base64
# from io import BytesIO

# import pandas as pd
# import joblib

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, precision_score, recall_score,
#     confusion_matrix,
#     mean_squared_error, mean_absolute_error, r2_score
# )

# # =====================================================
# # APP SETUP
# # =====================================================
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DATA_DIR = "data/datasets"
# os.makedirs(BASE_DATA_DIR, exist_ok=True)

# # =====================================================
# # UTILS
# # =====================================================
# def dataset_dir(dataset_id: str) -> str:
#     return os.path.join(BASE_DATA_DIR, dataset_id)

# def load_df(dataset_id: str) -> pd.DataFrame:
#     path = os.path.join(dataset_dir(dataset_id), "raw.csv")
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Dataset not found")
#     return pd.read_csv(path)

# def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     for col in df.columns:
#         if df[col].dtype == "object":
#             df[col] = df[col].astype(str).str.strip()
#         try:
#             df[col] = pd.to_numeric(df[col])
#         except Exception:
#             pass
#     return df

# def safe_json(val):
#     if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
#         return None
#     return val

# # =====================================================
# # UPLOAD (DAY-1)
# # =====================================================
# @app.post("/upload")
# async def upload_csv(file: UploadFile = File(...)):
#     dataset_id = file.filename.replace(".", "_")
#     dpath = dataset_dir(dataset_id)
#     os.makedirs(dpath, exist_ok=True)

#     if file.filename.endswith((".xlsx", ".xls")):
#         df = pd.read_excel(file.file)
#     else:
#         df = pd.read_csv(file.file)

#     df = clean_dataframe(df)
#     df.to_csv(os.path.join(dpath, "raw.csv"), index=False)

#     schema = {c: str(df[c].dtype) for c in df.columns}
#     json.dump(schema, open(os.path.join(dpath, "schema.json"), "w"), indent=2)

#     return {
#         "status": "ok",
#         "dataset_id": dataset_id,
#         "columns": list(df.columns),
#         "preview": df.head(5).to_dict(orient="records")
#     }

# # =====================================================
# # SCHEMA
# # =====================================================
# @app.get("/schema")
# def get_schema(dataset_id: str):
#     path = os.path.join(dataset_dir(dataset_id), "schema.json")
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Schema not found")
#     return {"status": "ok", "schema": json.load(open(path))}

# # =====================================================
# # META
# # =====================================================
# @app.get("/meta")
# def get_meta(dataset_id: str):
#     path = os.path.join(dataset_dir(dataset_id), "meta.json")
#     if not os.path.exists(path):
#         raise HTTPException(status_code=404, detail="Train the model first")
#     return json.load(open(path))

# # =====================================================
# # EDA (DAY-2/3/4)
# # =====================================================
# @app.get("/eda")
# def run_eda(dataset_id: str):
#     df = clean_dataframe(load_df(dataset_id))

#     num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

#     before_summary, after_summary = {}, {}
#     plots_before, plots_after, outliers = {}, {}, {}

#     for col in num_cols:
#         s = df[col].dropna()
#         if s.empty:
#             continue

#         before_summary[col] = s.describe().to_dict()

#         q1, q3 = s.quantile(0.25), s.quantile(0.75)
#         iqr = q3 - q1
#         low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr

#         outliers[col] = {
#             "count": int(((s < low) | (s > high)).sum()),
#             "lower": float(low),
#             "upper": float(high)
#         }

#         fig, ax = plt.subplots()
#         s.hist(ax=ax)
#         buf = BytesIO()
#         plt.savefig(buf, format="png")
#         plt.close()
#         plots_before[col] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

#         clipped = s.clip(low, high)
#         after_summary[col] = clipped.describe().to_dict()

#         fig, ax = plt.subplots()
#         clipped.hist(ax=ax)
#         buf = BytesIO()
#         plt.savefig(buf, format="png")
#         plt.close()
#         plots_after[col] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

#     return {
#         "status": "ok",
#         "eda": {
#             "missing": df.isnull().sum().to_dict(),
#             "dtypes": {c: str(df[c].dtype) for c in df.columns},
#             "before": {
#                 "summary": before_summary,
#                 "outliers": outliers,
#                 "plots": plots_before
#             },
#             "after": {
#                 "summary": after_summary,
#                 "plots": plots_after
#             }
#         }
#     }

# # =====================================================
# # TRAIN + EVALUATE (DAY-5 → DAY-9)
# # =====================================================
# @app.post("/train")
# def train_model(dataset_id: str, target: str):
#     df = clean_dataframe(load_df(dataset_id))

#     # ---------------- VALIDATION (DAY-8 CORE) ----------------
#     if target not in df.columns:
#         raise HTTPException(status_code=400, detail="Invalid target column")

#     # Reject ID-like or useless targets
#     if df[target].nunique() <= 1:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Target `{target}` has no variability"
#         )

#     # Drop rows where target is missing
#     df = df.dropna(subset=[target])

#     if df.shape[0] < 50:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Not enough data to train model for `{target}`"
#         )

#     X = df.drop(columns=[target])
#     y = df[target]

#     # Drop known bad columns
#     DROP_COLS = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "GarageYrBlt"]
#     X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

#     num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
#     cat_cols = [c for c in cat_cols if X[c].nunique() <= 20]

#     for c in cat_cols:
#         X[c] = X[c].astype(str)

#     features = num_cols + cat_cols

#     preprocessor = ColumnTransformer([
#         ("num", SimpleImputer(strategy="median"), num_cols),
#         ("cat", Pipeline([
#             ("imputer", SimpleImputer(strategy="most_frequent")),
#             ("encoder", OneHotEncoder(handle_unknown="ignore"))
#         ]), cat_cols)
#     ])

#     # ---------------- TASK DETECTION ----------------
#     if y.dtype == "object" or y.nunique() <= 15:
#         task = "classification"
#         model = RandomForestClassifier(n_estimators=200, random_state=42)
#     else:
#         task = "regression"
#         model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)

#     pipeline = Pipeline([
#         ("prep", preprocessor),
#         ("model", model)
#     ])

#     # ---------------- TRAIN / VALIDATION SPLIT ----------------
#     X_train, X_val, y_train, y_val = train_test_split(
#         X[features], y, test_size=0.2, random_state=42
#     )

#     pipeline.fit(X_train, y_train)
#     preds = pipeline.predict(X_val)

#     # ---------------- METRICS (GUARANTEED) ----------------
#     if task == "classification":
#         metrics = {
#             "accuracy": accuracy_score(y_val, preds),
#             "precision": precision_score(y_val, preds, average="weighted", zero_division=0),
#             "recall": recall_score(y_val, preds, average="weighted", zero_division=0),
#             "confusion_matrix": confusion_matrix(y_val, preds).tolist()
#         }
#     else:
#         metrics = {
#             "rmse": mean_squared_error(y_val, preds, squared=False),
#             "mae": mean_absolute_error(y_val, preds),
#             "r2": r2_score(y_val, preds)
#         }

#     # ---------------- FEATURE IMPORTANCE ----------------
#     top_features = features[:6]
#     model_step = pipeline.named_steps["model"]

#     if hasattr(model_step, "feature_importances_"):
#         importances = model_step.feature_importances_
#         ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
#         top_features = [f for f, _ in ranked[:6]]

#     # ---------------- SAVE ----------------
#     dpath = dataset_dir(dataset_id)
#     joblib.dump(pipeline, os.path.join(dpath, "model.pkl"))

#     meta = {
#         "target": target,
#         "task": task,
#         "features": features,
#         "top_features": top_features,
#         "metrics": metrics
#     }

#     with open(os.path.join(dpath, "meta.json"), "w") as f:
#         json.dump(meta, f, indent=2)

#     return {
#         "status": "ok",
#         "task": task,
#         "metrics": metrics,
#         "top_features": top_features
#     }


# # =====================================================
# # PREDICT (DAY-6 / DAY-7)
# # =====================================================
# @app.post("/predict")
# def predict(dataset_id: str, input_data: dict):
#     meta = json.load(open(os.path.join(dataset_dir(dataset_id), "meta.json")))
#     model = joblib.load(os.path.join(dataset_dir(dataset_id), "model.pkl"))

#     for col in meta["features"]:
#         input_data.setdefault(col, None)

#     df = clean_dataframe(pd.DataFrame([input_data]))
#     pred = model.predict(df)

#     return {"status": "ok", "prediction": safe_json(pred[0])}
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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
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
# TARGET VIABILITY ANALYSIS (SIMPLE VERSION)
# =====================================================
def validate_target(series: pd.Series) -> str | None:
    n = len(series)
    if series.isnull().mean() > 0.3:
        return "Target has too many missing values (>30%)"
    if series.nunique() / n > 0.9:
        return "Target is almost unique (ID-like column)"
    if series.nunique() < 2:
        return "Target has no meaningful variation"
    return None

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
# SCHEMA + TARGET ANALYSIS
# =====================================================
@app.get("/schema")
def get_schema(dataset_id: str):
    df = load_df(dataset_id)

    schema = {}
    target_analysis = {}

    for col in df.columns:
        schema[col] = str(df[col].dtype)
        reason = validate_target(df[col])
        target_analysis[col] = {
            "valid": reason is None,
            "reasons": [reason] if reason else []
        }

    return {
        "status": "ok",
        "schema": schema,
        "target_analysis": target_analysis
    }

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

        fig, ax = plt.subplots(figsize=(6, 4))
        s.hist(ax=ax, bins=30)
        ax.set_title(f"Before: {col}")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        plt.close()
        plots_before[col] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

        clipped = s.clip(low, high)
        after_summary[col] = clipped.describe().to_dict()

        fig, ax = plt.subplots(figsize=(6, 4))
        clipped.hist(ax=ax, bins=30)
        ax.set_title(f"After Outlier Clipping: {col}")
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
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
# TRAIN + EVALUATE (DAY-5 → DAY-9) - With smart model selection, CV details, VERSIONING & EXPLAINABILITY
# =====================================================
@app.post("/train")
def train_model(dataset_id: str, target: str):
    try:
        # ---------------- LOAD DATA ----------------
        df = clean_dataframe(load_df(dataset_id))

        # ---------------- TARGET VALIDATION ----------------
        if target not in df.columns:
            return {
                "status": "failed",
                "stage": "target_validation",
                "error": "Selected target does not exist in dataset"
            }

        reason = validate_target(df[target])
        if reason:
            return {
                "status": "failed",
                "stage": "target_validation",
                "error": reason
            }

        # Drop rows where target is missing
        df = df.dropna(subset=[target])

        if df.shape[0] < 50:
            return {
                "status": "failed",
                "stage": "validation",
                "error": "Not enough data after removing missing target values (need ≥50 rows)"
            }

        X = df.drop(columns=[target])
        y = df[target]

        # ---------------- DROP USELESS COLUMNS ----------------
        DROP_COLS = ["Alley", "PoolQC", "Fence", "MiscFeature", "FireplaceQu", "GarageYrBlt"]
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns])

        # ---------------- SPLIT FEATURES ----------------
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        cat_cols = [c for c in cat_cols if X[c].nunique() <= 20]

        for c in cat_cols:
            X[c] = X[c].astype(str)

        features = num_cols + cat_cols

        if not features:
            return {
                "status": "failed",
                "stage": "feature_engineering",
                "error": "No usable features after preprocessing"
            }

        # ---------------- PREPROCESSOR ----------------
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
        else:
            task = "regression"

        # ---------------- MODEL SELECTION ----------------
        if task == "classification":
            if y.nunique() <= 5:
                model = LogisticRegression(max_iter=1000)
                model_reason = "Few classes → Logistic Regression (stable, interpretable)"
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    class_weight="balanced"
                )
                model_reason = "Many classes or potential imbalance → Random Forest"
        else:
            if len(X) < 1000:
                model = LinearRegression()
                model_reason = "Small dataset → Linear Regression (low variance)"
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    random_state=42
                )
                model_reason = "Non-linear patterns likely → Random Forest"

        # ---------------- PIPELINE ----------------
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        # ---------------- TRAIN / VALIDATE ----------------
        X_train, X_test, y_train, y_test = train_test_split(
            X[features], y, test_size=0.2, random_state=42,
            stratify=y if task == "classification" else None
        )

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # ---------------- HOLD-OUT METRICS ----------------
        if task == "classification":
            test_metrics = {
                "accuracy": float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, preds, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, preds).tolist()
            }
        else:
            test_metrics = {
                "rmse": float(mean_squared_error(y_test, preds, squared=False)),
                "mae": float(mean_absolute_error(y_test, preds)),
                "r2": float(r2_score(y_test, preds))
            }

        # ---------------- CROSS VALIDATION WITH DETAILED SCORES ----------------
        cv_scoring = "accuracy" if task == "classification" else "r2"
        cv_scores = cross_val_score(
            pipeline,
            X[features],
            y,
            cv=5,
            scoring=cv_scoring
        )

        cv_details = {
            "scores": cv_scores.tolist(),
            "mean": float(cv_scores.mean()),
            "std": float(cv_scores.std())
        }

        # Combine all metrics
        metrics = {**test_metrics}
        metrics["cv"] = cv_details  # NEW: full CV details including per-fold scores

        # ---------------- FEATURE IMPORTANCE ----------------
        top_features = features[:6]
        model_step = pipeline.named_steps["model"]
        ranked = []  # default empty
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_
            ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            top_features = [f for f, _ in ranked[:6]]

        # ---- Explainability payload ----
        explainability = []
        if ranked:
            for fname, imp in ranked[:6]:
                explainability.append({
                    "feature": fname.split("_")[0],  # base feature name
                    "importance": round(float(imp), 4)
                })

        # ---------------- SAVE WITH VERSIONING ----------------
        dpath = dataset_dir(dataset_id)
        meta_path = os.path.join(dpath, "meta.json")

        # Determine next version number
        version = 1
        if os.path.exists(meta_path):
            try:
                old_meta = json.load(open(meta_path))
                version = old_meta.get("current_version", 0) + 1
            except Exception:
                version = 1  # fallback if meta corrupted

        model_filename = f"model_v{version}.pkl"
        model_path = os.path.join(dpath, model_filename)
        joblib.dump(pipeline, model_path)

        # Save updated meta with version info and explainability
        meta = {
            "target": target,
            "task": task,
            "features": features,
            "top_features": top_features,
            "best_model": model.__class__.__name__,
            "model_reason": model_reason,
            "current_version": version,
            "model_file": model_filename,
            "metrics": metrics,
            "explainability": explainability
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=safe_json)

        return {
            "status": "ok",
            "task": task,
            "best_model": meta["best_model"],
            "model_reason": model_reason,
            "current_version": version,
            "model_file": model_filename,
            "metrics": metrics,
            "top_features": top_features,
            "explainability": explainability
        }

    except ValueError as e:
        return {
            "status": "failed",
            "stage": "validation",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "failed",
            "stage": "unknown",
            "error": f"Training failed: {str(e)}"
        }

# =====================================================
# PREDICT (DAY-6 / DAY-7) - Updated to use versioned model
# =====================================================
@app.post("/predict")
def predict(dataset_id: str, input_data: dict):
    meta_path = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="Model not trained yet")

    meta = json.load(open(meta_path))
    model_filename = meta.get("model_file", "model_v1.pkl")  # fallback for old versions
    model_path = os.path.join(dataset_dir(dataset_id), model_filename)

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    model = joblib.load(model_path)

    # Ensure all required features are present
    for col in meta["features"]:
        input_data.setdefault(col, None)

    df_input = clean_dataframe(pd.DataFrame([input_data]))
    pred = model.predict(df_input)[0]

    return {"status": "ok", "prediction": safe_json(pred)}

# =====================================================
# CHAT WITH DATA
# =====================================================
@app.post("/chat")
def chat_with_data(dataset_id: str, question: str):
    meta_path = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(meta_path):
        return {
            "status": "failed",
            "error": "Train a model before asking questions"
        }

    df = load_df(dataset_id)
    meta = json.load(open(meta_path))

    q = question.lower()

    # ---- Rule-based intelligence (no LLM yet) ----
    if "sale price" in q and "sale type" in q:
        summary = (
            df.groupby("SaleType")["SalePrice"]
            .mean()
            .sort_values(ascending=False)
            .to_dict()
        )
        return {
            "answer": "Average SalePrice by SaleType",
            "data": summary
        }

    if "accuracy" in q or "performance" in q:
        return {
            "answer": "Model performance summary",
            "data": meta["metrics"]
        }

    if "features" in q or "important" in q:
        return {
            "answer": "Top influencing features",
            "data": meta["top_features"]
        }

    return {
        "answer": "I can answer questions about top features, model performance (including CV), and model choice reason."
    }