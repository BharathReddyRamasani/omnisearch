import time
import threading
import uuid
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json

# Global job registry
JOBS: Dict[str, Dict[str, Any]] = {}
executor = ThreadPoolExecutor(max_workers=3)  # Limit concurrent training jobs

def create_job():
    """Create a new background job"""
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "progress": 0,
        "message": "Job queued",
        "result": None,
        "error": None
    }
    return job_id

def update_job(job_id: str, status: str, progress: int = None, message: str = None, 
               result: Any = None, error: str = None):
    """Update job status"""
    if job_id not in JOBS:
        return
    
    JOBS[job_id]["status"] = status
    JOBS[job_id]["updated_at"] = time.time()
    
    if progress is not None:
        JOBS[job_id]["progress"] = progress
    
    if message is not None:
        JOBS[job_id]["message"] = message
    
    if result is not None:
        JOBS[job_id]["result"] = result
    
    if error is not None:
        JOBS[job_id]["error"] = error

def get_job(job_id: str) -> Dict[str, Any]:
    """Get job status"""
    return JOBS.get(job_id, {"status": "unknown"})

def submit_training_job(dataset_id: str, target: str, **kwargs):
    """Submit a training job to run in background"""
    job_id = create_job()
    
    def run_training():
        try:
            update_job(job_id, "running", 10, "Initializing training...")
            
            # Import here to avoid circular imports
            from backend.services.training import train_model_logic
            
            update_job(job_id, "running", 20, "Loading dataset...")
            result = train_model_logic(dataset_id, target, **kwargs)
            
            update_job(job_id, "running", 90, "Finalizing model...")
            
            if result["status"] == "ok":
                update_job(job_id, "completed", 100, "Training completed successfully", result=result)
            else:
                update_job(job_id, "failed", 100, f"Training failed: {result.get('error', 'Unknown error')}", error=result.get("error"))
                
        except Exception as e:
            update_job(job_id, "failed", 100, f"Training failed: {str(e)}", error=str(e))
    
    # Submit to thread pool
    executor.submit(run_training)
    
    return job_id

def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old completed/failed jobs"""
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    to_remove = []
    for job_id, job in JOBS.items():
        if job["status"] in ["completed", "failed"]:
            age = current_time - job.get("updated_at", job["created_at"])
            if age > max_age_seconds:
                to_remove.append(job_id)
    
    for job_id in to_remove:
        del JOBS[job_id]
