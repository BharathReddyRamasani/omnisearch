# backend/services/predict.py
import os
import joblib
import pandas as pd
from backend.services.utils import dataset_dir
from backend.services.utils import dataset_dir


def make_prediction(dataset_id: str, payload: dict):
    dpath = dataset_dir(dataset_id)
    model_path = os.path.join(dpath, "model.pkl")

    if not os.path.exists(model_path):
        return {"status": "failed", "error": "Model not trained"}

    model = joblib.load(model_path)
    df = pd.DataFrame([payload])

    pred = model.predict(df)[0]
    return {"status": "ok", "prediction": float(pred)}
