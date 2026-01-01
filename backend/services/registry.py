import time

JOBS = {}

def create_job():
    job_id = str(int(time.time() * 1000))
    JOBS[job_id] = {"status": "queued"}
    return job_id

def update_job(job_id, status, result=None):
    JOBS[job_id]["status"] = status
    if result is not None:
        JOBS[job_id]["result"] = result

def get_job(job_id):
    return JOBS.get(job_id, {"status": "unknown"})
