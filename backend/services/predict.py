# backend/services/predict.py
import os
import joblib
import pandas as pd
from backend.services.utils import model_dir


def make_prediction(dataset_id: str, payload: dict):
    model_path = os.path.join(model_dir(dataset_id), "model.pkl")

    if not os.path.exists(model_path):
        return {"status": "failed", "error": "Model not trained"}

    model = joblib.load(model_path)
    df = pd.DataFrame([payload])

    pred = model.predict(df)[0]
    conf = None
    if hasattr(model.named_steps["model"], "predict_proba"):
        conf = float(model.predict_proba(df)[0].max())

    return {"status": "ok", "prediction": pred, "confidence": conf}
