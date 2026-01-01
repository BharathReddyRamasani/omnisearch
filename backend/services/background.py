from threading import Thread
from backend.services.registry import update_job
from backend.services.training import train_model_logic

def run_training_job(job_id: str, dataset_id: str, target: str):
    try:
        update_job(job_id, "running")
        result = train_model_logic(dataset_id, target)
        update_job(job_id, "completed", result)
    except Exception as e:
        update_job(job_id, "failed", {"error": str(e)})

def start_training(job_id, dataset_id, target):
    Thread(
        target=run_training_job,
        args=(job_id, dataset_id, target),
        daemon=True
    ).start()
