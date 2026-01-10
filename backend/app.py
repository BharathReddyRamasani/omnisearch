from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import os
import json
import pandas as pd

from backend.services.ingest import process_upload
from backend.services.eda import generate_eda
from backend.services.cleaning import full_etl
from backend.services.training import train_model_logic
from backend.services.predict import make_prediction
from backend.services.utils import safe, raw_path, datasetdir, model_dir

# =====================================================
# APP
# =====================================================
app = FastAPI(title="OmniSearch AI – Enterprise AutoML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# UPLOAD
# =====================================================
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    result = await process_upload(file)
    preview = pd.read_csv(raw_path(result["dataset_id"]), nrows=5)

    return safe({
        "status": "ok",
        "dataset_id": result["dataset_id"],
        "columns": preview.columns.tolist(),
        "preview": preview.to_dict("records"),
    })

# =====================================================
# EDA
# =====================================================
@app.get("/eda/{dataset_id}")
def eda(dataset_id: str):
    return safe(generate_eda(dataset_id))

# =====================================================
# ETL / CLEAN
# =====================================================
@app.post("/datasets/{dataset_id}/clean")
def run_etl(dataset_id: str):
    result = full_etl(dataset_id)
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "ETL failed"))
    return safe(result)

# =====================================================
# DOWNLOADS
# =====================================================
@app.get("/datasets/{dataset_id}/download/{kind}")
def download(dataset_id: str, kind: str):
    if kind == "raw":
        p = raw_path(dataset_id)
    elif kind == "clean":
        p = os.path.join(datasetdir(dataset_id), "clean.csv")
    else:
        raise HTTPException(400, "Invalid download type: raw or clean only")

    if not os.path.exists(p):
        raise HTTPException(404, f"{kind.capitalize()} file not found")

    return FileResponse(p, filename=f"{dataset_id}_{kind}.csv")

# =====================================================
# COMPARISON (RAW vs CLEAN)
# =====================================================
@app.get("/datasets/{dataset_id}/comparison")
def comparison(dataset_id: str):
    p = os.path.join(datasetdir(dataset_id), "comparison.json")
    if not os.path.exists(p):
        raise HTTPException(404, "Run ETL/cleaning first")
    return safe(json.load(open(p)))

# =====================================================
# TRAIN
# =====================================================
@app.post("/train/{dataset_id}")
def train(dataset_id: str, payload: dict):
    target = payload.get("target")
    if not target:
        raise HTTPException(400, "Target column name required in payload")
    
    # Extract optional parameters with defaults
    test_size = payload.get("test_size", 0.2)
    random_state = payload.get("random_state", 42)
    train_regression = payload.get("train_regression", True)
    train_classification = payload.get("train_classification", True)

    result = train_model_logic(dataset_id, target, test_size, random_state, train_regression, train_classification)
    if result["status"] != "ok":
        raise HTTPException(400, result["error"])
    return safe(result)

# =====================================================
# META
# =====================================================
@app.get("/meta/{dataset_id}")
def meta(dataset_id: str):
    p = os.path.join(model_dir(dataset_id), "metadata.json")
    if not os.path.exists(p):
        raise HTTPException(404, "No model trained yet for this dataset")
    return safe(json.load(open(p)))

# =====================================================
# PREDICT
# =====================================================
@app.post("/predict/{dataset_id}")
def predict(dataset_id: str, payload: dict):
    result = make_prediction(dataset_id, payload)
    if result["status"] != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    return safe(result)

# =====================================================
# CHAT (DSL-based with RAG)
# =====================================================
# from backend.services.chat import get_chat_response

# @app.post("/chat/{dataset_id}")
# def chat(dataset_id: str, payload: dict):
#     question = payload.get("question", "")
#     history = payload.get("history", [])
#     if not question:
#         raise HTTPException(status_code=400, detail="Question required")
#     result = get_chat_response(dataset_id, question, history)
#     return safe(result)