import os
import json
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from backend.services.utils import load_raw, model_dir


def train_model_logic(dataset_id: str, target: str):
    df = load_raw(dataset_id)
    if target not in df.columns:
        return {"status": "failed", "error": "Invalid target"}

    X = df.drop(columns=[target])
    y = df[target]
    mask = ~y.isna()
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return {"status": "failed", "error": "Not enough valid rows"}

    task = "classification" if y.nunique() <= 15 else "regression"

    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(include="object").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ])

    if task == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(n_estimators=200),
            "GradientBoosting": GradientBoostingClassifier(),
        }
        scorer = lambda yt, yp: accuracy_score(yt, yp)
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(n_estimators=200),
            "GradientBoosting": GradientBoostingRegressor(),
        }
        scorer = lambda yt, yp: r2_score(yt, yp)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    leaderboard = []
    best_score = -1e9
    best_model = None
    best_pipe = None

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        score = scorer(yte, preds)

        leaderboard.append({
            "model": name,
            "score": round(float(score), 4),
            "train_rows": len(Xtr),
            "test_rows": len(Xte)
        })

        if score > best_score:
            best_score = score
            best_model = name
            best_pipe = pipe

    root = model_dir(dataset_id)
    joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

    result = {
        "status": "ok",
        "task": task,
        "target": target,
        "best_model": best_model,
        "best_score": round(float(best_score), 4),
        "leaderboard": leaderboard,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "explanation": {
            "why_best": f"{best_model} achieved the highest validation score.",
            "why_not_others": "Other models underperformed on the same dataset and split."
        }
    }

    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
