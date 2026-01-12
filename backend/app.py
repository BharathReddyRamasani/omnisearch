from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


import os
import json
import pandas as pd

from backend.services.ingest import process_upload
from backend.services.eda import generate_eda
from backend.services.cleaning import full_etl
from backend.services.training import train_model_logic
from backend.services.predict import make_prediction, make_batch_prediction
from backend.services.model_registry import get_active_model_metadata, get_model_history
from backend.services.registry import submit_training_job, get_job
from backend.services.utils import safe, raw_path, datasetdir, model_dir
from backend.services.chat import get_chat_response



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
# API ROUTER
# =====================================================
api_router = APIRouter(prefix="/api")

# =====================================================
# TEST ROUTE
# =====================================================
@api_router.get("/test")
def test():
    return {"status": "ok", "message": "API is working"}

# =====================================================
# UPLOAD
# =====================================================
@api_router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    try:
        result = await process_upload(file)
        
        # Generate preview safely
        raw_file_path = raw_path(result["dataset_id"])
        if os.path.exists(raw_file_path):
            try:
                preview_df = pd.read_csv(raw_file_path, nrows=5)
                preview = preview_df.to_dict("records")
            except Exception as e:
                # If preview fails, use the one from process_upload
                preview = result.get("preview", [])
        else:
            preview = result.get("preview", [])

        return safe({
            "status": "ok",
            "dataset_id": result["dataset_id"],
            "columns": result["columns"],
            "preview": preview,
        })
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

# =====================================================
# EDA
# =====================================================
@api_router.get("/eda/{dataset_id}")
def eda(dataset_id: str):
    return safe(generate_eda(dataset_id))

# =====================================================
# ETL / CLEAN
# =====================================================
@api_router.post("/datasets/{dataset_id}/clean")
def run_etl(dataset_id: str):
    result = full_etl(dataset_id)
    if result.get("status") != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "ETL failed"))
    return safe(result)

# =====================================================
# DOWNLOADS
# =====================================================
@api_router.get("/datasets/{dataset_id}/download/{kind}")
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
@api_router.get("/datasets/{dataset_id}/comparison")
def comparison(dataset_id: str):
    p = os.path.join(datasetdir(dataset_id), "comparison.json")
    if not os.path.exists(p):
        # Return empty comparison instead of 404
        return safe({
            "status": "info",
            "message": "ETL not run yet. Use /datasets/{dataset_id}/clean to run ETL.",
            "comparison": {
                "raw_stats": {},
                "clean_stats": {},
                "improvements": {}
            }
        })
    try:
        with open(p, 'r') as f:
            data = json.load(f)
            # Ensure it's wrapped with comparison key for consistency
            if "comparison" not in data:
                data = {"comparison": data}
            return safe(data)
    except Exception as e:
        raise HTTPException(500, f"Error reading comparison: {str(e)}")

# =====================================================
# DATASET INFO & SAMPLE
# =====================================================
@api_router.get("/datasets/{dataset_id}/info")
def dataset_info(dataset_id: str):
    """Get basic dataset information"""
    try:
        # Try clean first, fallback to raw
        clean_path = os.path.join(datasetdir(dataset_id), "clean.csv")
        raw_path_file = raw_path(dataset_id)
        
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path)
            source = "clean"
        elif os.path.exists(raw_path_file):
            df = pd.read_csv(raw_path_file)
            source = "raw"
        else:
            raise HTTPException(404, "Dataset not found")
        
        return safe({
            "status": "ok",
            "dataset_id": dataset_id,
            "source": source,
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "missing_values": df.isnull().sum().to_dict()
        })
    except Exception as e:
        raise HTTPException(500, f"Error getting dataset info: {str(e)}")

@api_router.get("/datasets/{dataset_id}/sample")
def dataset_sample(dataset_id: str, nrows: int = 5):
    """Get sample rows from dataset (clean if available, else raw)"""
    try:
        clean_path = os.path.join(datasetdir(dataset_id), "clean.csv")
        raw_path_file = raw_path(dataset_id)
        
        if os.path.exists(clean_path):
            df = pd.read_csv(clean_path, nrows=nrows)
            source = "clean"
        elif os.path.exists(raw_path_file):
            df = pd.read_csv(raw_path_file, nrows=nrows)
            source = "raw"
        else:
            raise HTTPException(404, "Dataset not found")
        
        return safe({
            "status": "ok",
            "dataset_id": dataset_id,
            "source": source,
            "data": df.to_dict("records")
        })
    except Exception as e:
        raise HTTPException(500, f"Error getting sample: {str(e)}")

# =====================================================
# TRAIN (BACKGROUND ASYNC)
# =====================================================
@api_router.post("/train/{dataset_id}")
def train(dataset_id: str, payload: dict):
    target = payload.get("target")
    if not target:
        raise HTTPException(400, "Target column name required in payload")
    
    # Extract optional parameters with defaults
    test_size = payload.get("test_size", 0.2)
    random_state = payload.get("random_state", 42)
    train_regression = payload.get("train_regression", True)
    train_classification = payload.get("train_classification", True)
    time_limit_seconds = payload.get("time_limit_seconds")

    # Submit background job
    job_id = submit_training_job(
        dataset_id=dataset_id,
        target=target,
        test_size=test_size,
        random_state=random_state,
        train_regression=train_regression,
        train_classification=train_classification,
        time_limit_seconds=time_limit_seconds
    )
    
    return safe({
        "status": "accepted",
        "job_id": job_id,
        "message": "Training job submitted. Check status with /jobs/{job_id}"
    })

# =====================================================
# JOB STATUS
# =====================================================
@api_router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = get_job(job_id)
    if job["status"] == "unknown":
        raise HTTPException(404, "Job not found")
    
    return safe(job)

# =====================================================
# TRAIN (SYNCHRONOUS - LEGACY)
# =====================================================
@api_router.post("/train/{dataset_id}/sync")
def train_sync(dataset_id: str, payload: dict):
    """Synchronous training for small datasets or testing"""
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
# META (MODEL METADATA VIA REGISTRY)
# =====================================================
@api_router.get("/meta/{dataset_id}")
def meta(dataset_id: str):
    metadata = get_active_model_metadata(dataset_id)
    if not metadata:
        raise HTTPException(404, "No active model found for this dataset")
    return safe(metadata)

# =====================================================
# MODEL HISTORY
# =====================================================
@api_router.get("/models/{dataset_id}/history")
def model_history(dataset_id: str):
    history = get_model_history(dataset_id)
    if not history:
        raise HTTPException(404, "No model history found for this dataset")
    return safe({"history": history})

# =====================================================
# MODEL ROLLBACK
# =====================================================
@api_router.post("/models/{dataset_id}/rollback")
def rollback_model_version(dataset_id: str, payload: dict):
    version_id = payload.get("version_id")
    if not version_id:
        raise HTTPException(400, "version_id required")
    
    from backend.services.model_registry import rollback_model
    result = rollback_model(dataset_id, version_id)
    if result["status"] != "ok":
        raise HTTPException(400, result["error"])
    return safe(result)

# =====================================================
# PREDICT
# =====================================================
@api_router.post("/predict/{dataset_id}")
def predict(dataset_id: str, payload: dict):
    result = make_prediction(dataset_id, payload)
    if result["status"] != "ok":
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    return safe(result)

# =====================================================
# BATCH PREDICT
# =====================================================
@api_router.post("/predict/{dataset_id}/batch")
async def predict_batch(dataset_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed for batch prediction")

    content = await file.read()
    result = make_batch_prediction(dataset_id, content)
    
    if result["status"] != "ok":
        raise HTTPException(400, result.get("error", "Batch prediction failed"))
    
    # Return both JSON response and CSV download
    return safe(result)

# =====================================================
# CHAT (DSL-based with RAG)
# =====================================================
# from backend.services.chat import get_chat_response

@api_router.post("/chat/{dataset_id}")
def chat(dataset_id: str, payload: dict):
    question = payload.get("question", "")
    history = payload.get("history", [])
    if not question:
        raise HTTPException(status_code=400, detail="Question required")
    result = get_chat_response(dataset_id, question, history)
    return safe(result)

# Include the API router
app.include_router(api_router)
# =====================================================
# DATASET INFO & SAMPLE



