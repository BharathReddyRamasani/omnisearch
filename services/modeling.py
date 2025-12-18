import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from services.features import build_preprocessor

def train_model(df: pd.DataFrame, target: str, dataset_id: str):
    df = df.copy()
    df[target] = df[target].replace(" ", pd.NA)
    df = df.dropna(subset=[target])

    if len(df) < 20:
        return {"error": "Not enough data after cleaning target"}

    X = df.drop(columns=[target])
    y = df[target]

    if y.nunique() > 0.9 * len(y):
        return {"error": "Target looks like ID column"}

    if y.dtype == "object" and y.nunique() <= 20:
        problem = "classification"
        model = RandomForestClassifier(random_state=42)
    elif pd.api.types.is_numeric_dtype(y):
        problem = "regression"
        model = RandomForestRegressor(random_state=42)
    else:
        return {"error": "Unsupported target type"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if problem == "classification":
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred, average="weighted"), 4)
        }
    else:
        metrics = {
            "r2": round(r2_score(y_test, y_pred), 4),
            "rmse": round(mean_squared_error(y_test, y_pred, squared=False), 4)
        }

    base = f"data/datasets/{dataset_id}"
    joblib.dump(pipeline, f"{base}/model.pkl")

    with open(f"{base}/schema.json", "w") as f:
        json.dump(list(X.columns), f)

    return {
        "problem_type": problem,
        "metrics": metrics,
        "model_saved": True
    }
