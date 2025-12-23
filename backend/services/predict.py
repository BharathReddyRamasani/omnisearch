import os
import json
import joblib
import pandas as pd
from backend.services.utils import dataset_dir

def make_prediction(dataset_id: str, payload: dict):
    base = f"data/datasets/{dataset_id}"
    model = joblib.load(f"{base}/model_v1.pkl")  # Use latest version
    
    with open(f"{base}/schema.json") as f:
        cols = json.load(f)
    
    df = pd.DataFrame([payload])
    for c in cols:
        if c not in df:
            df[c] = None
    
    df = df[cols]
    pred = model.predict(df)[0]
    
    return {"prediction": pred}
