import os
import json
import joblib
import pandas as pd

def run_prediction(dataset_id: str, payload: dict):
    base = f"data/datasets/{dataset_id}"

    model = joblib.load(f"{base}/model.pkl")

    with open(f"{base}/schema.json") as f:
        cols = json.load(f)

    df = pd.DataFrame([payload])

    for c in cols:
        if c not in df:
            df[c] = None

    df = df[cols]
    pred = model.predict(df)

    return pred[0]
