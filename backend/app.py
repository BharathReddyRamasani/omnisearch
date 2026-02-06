from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.requests import Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException

import os
import json
import pandas as pd
import logging

# All service imports together
from backend.services.ingest import process_upload
from backend.services.eda import generate_eda
from backend.services.cluster import run_clustering
from backend.services.anomaly import run_anomaly_detection
from backend.services.advanced import run_pca, run_tsne
from backend.services.cleaning import full_etl
from backend.services.training import train_model_logic
from backend.services.predict import make_prediction, make_batch_prediction
from backend.services.model_registry import get_active_model_metadata, get_model_history
from backend.services.registry import submit_training_job, get_job
from backend.services.utils import safe, raw_path, datasetdir, model_dir
from backend.services.chat import get_chat_response

# =====================================================
# LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
# EXCEPTION HANDLERS (HTTPException MUST come FIRST)
# =====================================================

# Handle FastAPI HTTPExceptions with proper status codes
@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc: FastAPIHTTPException):
    """
    Handle FastAPI HTTPExceptions.
    
    Must be defined BEFORE general Exception handler to take priority.
    Preserves the original status code and structured error format.
    """
    logger.warning(f"HTTP exception {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": exc.detail,
            "request_path": request.url.path
        }
    )


# Handle all other unhandled exceptions (fallback)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Centralized exception handler for all unhandled exceptions (non-HTTP).
    
    Only catches exceptions that are NOT HTTPException.
    Provides consistent error response format with:
    - User-friendly message
    - Error type for debugging
    - Request context logging
    """
    error_type = type(exc).__name__
    error_message = str(exc)
    
    logger.error(f"Unhandled exception: {error_type}: {error_message}", exc_info=exc)
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": error_type,
            "message": error_message if error_message else "Internal server error",
            "request_path": request.url.path
        }
    )

# =====================================================
# API ROUTER
# =====================================================
api_router = APIRouter(prefix="/api")

# =====================================================
# HELPER: MAPPING CONFIRMATION CHECKER
# =====================================================
def check_mapping_confirmed(dataset_id: str) -> dict:
    """
    Verify that column mapping has been confirmed for a dataset.
    
    Loads upload_metadata.json and checks column_mapping_confirmed flag.
    
    Raises HTTPException(400) if not confirmed.
    
    Returns metadata dict if confirmed.
    """
    try:
        dpath = datasetdir(dataset_id)
        metadata_path = os.path.join(dpath, "upload_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise HTTPException(
                404,
                f"Upload metadata not found for dataset {dataset_id}. Please upload a dataset first."
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check confirmation status
        if not metadata.get("column_mapping_confirmed", False):
            raise HTTPException(
                400,
                "Column mapping not confirmed. Please confirm the mapping in the Upload page before proceeding."
            )
        
        return metadata
    
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error checking mapping confirmation for {dataset_id}: {str(e)}", exc_info=e)
        raise HTTPException(500, f"Error checking mapping confirmation: {str(e)}")

def load_data_for_eda(dataset_id: str) -> pd.DataFrame:
    """
    Load data for EDA, preferring clean data if available, falling back to ingested.
    
    This mirrors frontend behavior: if ETL is complete, use clean.csv, otherwise use ingested.csv
    (which has normalized column names)
    """
    from backend.services.utils import load_clean, load_ingested
    dpath = datasetdir(dataset_id)
    clean_path = os.path.join(dpath, "clean.csv")
    if os.path.exists(clean_path):
        return load_clean(dataset_id)
    else:
        return load_ingested(dataset_id)

# =====================================================
# EDA ADVANCED ANALYTICS ENDPOINTS
# =====================================================
@api_router.post("/eda/cluster/{dataset_id}")
def eda_cluster(dataset_id: str, payload: dict = Body(...)):
    """
    Run clustering on selected features and return labels and base64 plot.
    Payload: {"features": [...], "algo": "KMeans", "n_clusters": 3}
    """
    check_mapping_confirmed(dataset_id)
    df = load_data_for_eda(dataset_id)
    features = payload.get("features", [])
    algo = payload.get("algo", "KMeans")
    n_clusters = payload.get("n_clusters", 3)
    result = run_clustering(df, features, algo, n_clusters)
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error", "Clustering failed"))
    return {"status": "ok", **result}

@api_router.post("/eda/anomaly/{dataset_id}")
def eda_anomaly(dataset_id: str, payload: dict = Body(...)):
    """
    Run anomaly detection on selected features and return labels and base64 plot.
    Payload: {"features": [...], "method": "Isolation Forest", "contamination": 0.05}
    """
    check_mapping_confirmed(dataset_id)
    df = load_data_for_eda(dataset_id)
    features = payload.get("features", [])
    method = payload.get("method", "Isolation Forest")
    contamination = payload.get("contamination", 0.05)
    result = run_anomaly_detection(df, features, method, contamination)
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error", "Anomaly detection failed"))
    return {"status": "ok", **result}

@api_router.post("/eda/advanced/{dataset_id}")
def eda_advanced(dataset_id: str, payload: dict = Body(...)):
    """
    Run advanced EDA (PCA, t-SNE) on selected features.
    Payload: {"features": [...], "method": "PCA" or "t-SNE", "n_components": 2}
    """
    check_mapping_confirmed(dataset_id)
    df = load_data_for_eda(dataset_id)
    features = payload.get("features", [])
    method = payload.get("method", "PCA")
    n_components = payload.get("n_components", 2)
    if method == "PCA":
        result = run_pca(df, features, n_components)
    elif method == "t-SNE":
        result = run_tsne(df, features, n_components)
    else:
        raise HTTPException(400, "Unknown advanced EDA method")
    if result.get("status") == "failed":
        raise HTTPException(status_code=400, detail=result.get("error", f"{method} failed"))
    return {"status": "ok", **result}

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
    """
    Upload and process a CSV file with industrial-grade validation.
    
    Supports:
    - Encoding detection (charset-normalizer)
    - Column normalization with mapping
    - Type inference and coercion
    - File size validation (500MB max)
    - Dimension validation (500 cols, 100k rows max)
    - Sample-based reading for memory efficiency
    
    Returns:
        {
            "status": "ok",
            "dataset_id": str,
            "rows": int,
            "is_sampled": bool,
            "sample_limit": int,
            "columns": [str],
            "column_mapping": {original: normalized},
            "encoding": {"detected": str, "confidence": float, "detection_method": str},
            "coercion_summary": {col: {type, coercions, method}},
            "preview": [dict]
        }
    """
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
            "rows": result.get("rows", 0),
            "is_sampled": result.get("is_sampled", False),
            "sample_limit": result.get("sample_limit", 100000),
            "columns": result["columns"],
            "column_mapping": result.get("column_mapping", {}),
            "encoding": result.get("encoding", {"detected": "utf-8", "confidence": 0, "detection_method": "unknown"}),
            "coercion_summary": result.get("coercion_summary", {}),
            "preview": preview,
        })
    except HTTPException as he:
        # Re-raise HTTP exceptions from ingest module
        raise he
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=e)
        raise HTTPException(500, f"Upload failed: {str(e)}")

# =====================================================
# UPLOAD CONFIRMATION (Column Mapping)
# =====================================================
@api_router.post("/datasets/{dataset_id}/confirm-mapping")
def confirm_column_mapping(dataset_id: str):
    """
    Record that user has confirmed the column name mappings.
    
    Updates upload_metadata.json with:
    - column_mapping_confirmed: true
    - confirmation_timestamp: ISO timestamp
    
    This creates an audit trail for data governance.
    """
    try:
        dpath = datasetdir(dataset_id)
        metadata_path = os.path.join(dpath, "upload_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise HTTPException(404, f"Upload metadata not found for dataset {dataset_id}")
        
        # Load existing metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update confirmation status
        metadata["column_mapping_confirmed"] = True
        metadata["confirmation_timestamp"] = pd.Timestamp.utcnow().isoformat()
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Column mapping confirmed for dataset {dataset_id}")
        
        return safe({
            "status": "ok",
            "dataset_id": dataset_id,
            "message": "Column mapping confirmed",
            "column_mapping_confirmed": True,
            "confirmation_timestamp": metadata["confirmation_timestamp"]
        })
    
    except Exception as e:
        logger.error(f"Failed to confirm mapping for {dataset_id}: {str(e)}", exc_info=e)
        raise HTTPException(500, f"Failed to confirm mapping: {str(e)}")

# =====================================================
# EDA
# =====================================================
@api_router.get("/eda/{dataset_id}")
def eda(dataset_id: str):
    """
    Generate exploratory data analysis for dataset.
    
    Requires: Column mapping must be confirmed first.
    """
    # Verify mapping confirmation (data governance)
    check_mapping_confirmed(dataset_id)
    
    return safe(generate_eda(dataset_id))

# =====================================================
# ETL / CLEAN
# =====================================================
@api_router.post("/datasets/{dataset_id}/clean")
def run_etl(dataset_id: str):
    """
    Run ETL pipeline to clean and transform dataset.
    
    Requires: Column mapping must be confirmed first.
    """
    # Verify mapping confirmation (data governance)
    check_mapping_confirmed(dataset_id)
    
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
    """
    Submit async training job for dataset.
    
    Requires: Column mapping must be confirmed first.
    """
    # Verify mapping confirmation (data governance)
    check_mapping_confirmed(dataset_id)
    
    target = payload.get("target")
    if not target:
        raise HTTPException(400, "Target column name required in payload")
    
    # Extract optional parameters with defaults
    test_size = payload.get("test_size", 0.2)
    random_state = payload.get("random_state", 42)
    task = payload.get("task")  # None = auto-detect, "classification", or "regression"
    time_limit_seconds = payload.get("time_limit_seconds")

    # Submit background job
    job_id = submit_training_job(
        dataset_id=dataset_id,
        target=target,
        test_size=test_size,
        random_state=random_state,
        task=task,
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
    """
    Synchronous training for small datasets or testing.
    
    Requires: Column mapping must be confirmed first.
    """
    # Verify mapping confirmation (data governance)
    check_mapping_confirmed(dataset_id)
    
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
    # Validate dataset_id
    if not dataset_id or not isinstance(dataset_id, str):
        raise HTTPException(status_code=400, detail="Invalid dataset_id")
    
    # Clean dataset_id (handle case where it has commas appended)
    dataset_id = str(dataset_id).strip()
    if "," in dataset_id:
        dataset_id = dataset_id.split(",")[0].strip()
    
    # Ensure payload is a dict
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a dictionary")
    
    result = make_prediction(dataset_id, payload)
    # ✅ Return all valid statuses - "ok" and "failed" should both be returned to client
    if result.get("status") not in ("ok", "failed"):
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
    return safe(result)

# =====================================================
# BATCH PREDICT
# =====================================================
@api_router.post("/predict/{dataset_id}/batch")
async def predict_batch(dataset_id: str, file: UploadFile = File(...)):
    # Validate dataset_id
    if not dataset_id or not isinstance(dataset_id, str):
        raise HTTPException(status_code=400, detail="Invalid dataset_id")
    
    # Clean dataset_id
    dataset_id = str(dataset_id).strip()
    if "," in dataset_id:
        dataset_id = dataset_id.split(",")[0].strip()
    
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed for batch prediction")

    content = await file.read()
    result = make_batch_prediction(dataset_id, content)
    
    if result.get("status") not in ("ok", "error"):
        raise HTTPException(400, result.get("error", "Batch prediction failed"))
    
    # Return both JSON response and CSV download
    return safe(result)

# =====================================================
# CHAT (DSL-based with RAG)
# =====================================================
from backend.services.chat import get_chat_response

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





