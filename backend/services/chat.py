import os
import json
from backend.services.utils import dataset_dir, load_df

def get_chat_response(dataset_id: str, question: str):
    metapath = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(metapath):
        return {"status": "failed", "error": "Train a model first"}
    
    meta = json.load(open(metapath))
    q = question.lower()
    
    if "accuracy" in q or "performance" in q:
        return {"answer": "Model Metrics:", "data": meta["metrics"]}
    
    if "features" in q or "important" in q:
        return {"answer": "Top Features:", "data": meta["top_features"]}
    
    if "model" in q:
        return {"answer": f"Used {meta['best_model']} because {meta['model_reason']}"}
    
    return {
        "answer": "I can answer questions about model performance, top features, and model selection."
    }
