# backend/background.py
import os
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from typing import Optional

from backend.services.registry import update_job
from backend.services.training import train_model_logic
import traceback
import time
import json
from threading import Thread

logger = logging.getLogger(__name__)

# Global executor — shared across requests, limited concurrency
_executor = ProcessPoolExecutor(
    max_workers=2,  # ← Adjust: 1–4 depending on CPU cores / RAM
    # initializer=some_init_fn_if_needed
)


def run_training_job(
    job_id: str,
    dataset_id: str,
    target: str,
    timeout_seconds: int = 1800  # 30 min default
) -> None:
    """
    Runs training in a separate PROCESS (not thread).
    Updates job registry on success/failure/timeout.
    """
    update_job(job_id, "running", {"message": "Training started in background process"})

    try:
        # Submit to process pool with timeout
        future = _executor.submit(train_model_logic, dataset_id, target)

        # Block here with timeout — this runs in FastAPI worker thread
        result = future.result(timeout=timeout_seconds)

        update_job(job_id, "completed", {
            "result": result,
            "message": "Training completed successfully",
            "completed_at": time.time()
        })

        logger.info(f"Job {job_id} completed successfully")

    except TimeoutError:
        # Attempt to kill — note: processes are hard to kill cleanly
        future.cancel()
        update_job(job_id, "failed", {
            "error": f"Training timed out after {timeout_seconds} seconds",
            "timeout_seconds": timeout_seconds
        })
        logger.warning(f"Job {job_id} timed out after {timeout_seconds}s")

    except Exception as e:
        update_job(job_id, "failed", {
            "error": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))
        })
        logger.exception(f"Job {job_id} failed during training")

    finally:
        # Optional: cleanup any temp files if needed
        pass


def start_training(
    job_id: str,
    dataset_id: str,
    target: str,
    timeout_seconds: Optional[int] = None
) -> None:
    """
    Fire-and-forget start of background training job.
    Non-blocking — returns immediately.
    """
    if timeout_seconds is None:
        timeout_seconds = 1800  # 30 min default

    # Start in background — does NOT block the FastAPI request
    Thread(
        target=run_training_job,
        args=(job_id, dataset_id, target, timeout_seconds),
        daemon=False   # ← CRITICAL: NOT daemon!
    ).start()

    logger.info(f"Started background training job {job_id} for dataset {dataset_id}")