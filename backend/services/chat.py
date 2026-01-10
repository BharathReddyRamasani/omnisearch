import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline
from backend.services.utils import datasetdir, load_raw, load_clean, model_dir
from backend.services.eda import generate_eda  # For summaries

# Global model caches
_embedding_model = None
_llm_pipeline = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast
    return _embedding_model

def get_llm_pipeline():
    global _llm_pipeline
    if _llm_pipeline is None:
        try:
            _llm_pipeline = pipeline("text-generation", model="distilgpt2", max_length=300, temperature=0.1)
        except Exception as e:
            # Fallback to a simpler approach if model fails to load
            print(f"Warning: Could not load LLM model: {e}")
            _llm_pipeline = None
    return _llm_pipeline

def get_comprehensive_context(dataset_id: str) -> List[str]:
    """Gather all available information about the dataset for RAG"""
    docs = []

    # Basic dataset info
    try:
        df_raw = load_raw(dataset_id)
        docs.append(f"Raw dataset has {len(df_raw)} rows and {len(df_raw.columns)} columns: {', '.join(df_raw.columns.tolist())}")
        docs.append(f"Raw data types: {df_raw.dtypes.to_dict()}")
        docs.append(f"Raw data sample: {df_raw.head(3).to_string()}")
    except Exception as e:
        docs.append(f"Raw data not available: {str(e)}")

    # Clean data info
    try:
        df_clean = load_clean(dataset_id)
        docs.append(f"Clean dataset has {len(df_clean)} rows and {len(df_clean.columns)} columns")
        docs.append(f"Clean data types: {df_clean.dtypes.to_dict()}")
        docs.append(f"Clean data sample: {df_clean.head(3).to_string()}")
    except:
        docs.append("Clean data not available - ETL not run yet")

    # EDA results
    try:
        eda_path = os.path.join(datasetdir(dataset_id), "eda.json")
        if os.path.exists(eda_path):
            eda = json.load(open(eda_path))
            docs.append(f"EDA Summary: {eda['eda']['rows']} rows, {eda['eda']['columns']} columns")
            docs.append(f"Data Quality: Score {eda['eda']['quality_score']}/100, Grade {eda['eda']['quality_grade']}")
            docs.append(f"Missing Data: {eda['eda']['missing_total']} total missing values")
            docs.append(f"Outlier Percentage: {eda['eda']['outlier_pct']}%")
            docs.append(f"Column Statistics: {json.dumps(eda['eda']['summary'], indent=2)}")
    except:
        docs.append("EDA not available")

    # ETL comparison
    try:
        comp_path = os.path.join(datasetdir(dataset_id), "comparison.json")
        if os.path.exists(comp_path):
            comp = json.load(open(comp_path))
            docs.append(f"ETL Results: {comp}")
    except:
        docs.append("ETL comparison not available")

    # Model metadata
    try:
        meta_path = os.path.join(model_dir(dataset_id), "metadata.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            docs.append(f"Trained Model: {meta}")
            docs.append(f"Model Performance: {meta.get('metrics', {})}")
            docs.append(f"Target Column: {meta.get('target', 'Unknown')}")
            docs.append(f"Feature Importance: {meta.get('feature_importance', {})}")
    except:
        docs.append("No trained model available")

    return docs

def create_rag_index(dataset_id: str):
    index_path = os.path.join(datasetdir(dataset_id), "faiss_index.idx")
    docs_path = os.path.join(datasetdir(dataset_id), "docs.json")

    if os.path.exists(index_path):
        return faiss.read_index(index_path), json.load(open(docs_path))

    # Create comprehensive docs
    docs = get_comprehensive_context(dataset_id)

    if not docs:
        docs = ["No information available for this dataset"]

    model = get_embedding_model()
    embeddings = model.encode(docs)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype('float32'))

    faiss.write_index(index, index_path)
    with open(docs_path, "w") as f:
        json.dump(docs, f)

    return index, docs

def retrieve_context(dataset_id: str, question: str, top_k=5):
    index, docs = create_rag_index(dataset_id)
    model = get_embedding_model()
    query_emb = model.encode([question])
    distances, indices = index.search(query_emb.astype('float32'), top_k)
    return [docs[i] for i in indices[0] if i < len(docs)]

def generate_industrial_response(question: str, context: List[str], dataset_id: str) -> str:
    """Generate comprehensive, mentor-like responses"""

    llm = get_llm_pipeline()
    if llm is None:
        # Fallback response without LLM
        context_text = "\n".join(context)
        return f"Based on the available data for dataset {dataset_id}:\n\n{context_text}\n\nFor your question '{question}', I recommend checking the model metadata and EDA results for detailed insights."

    context_text = "\n".join(context)

    system_prompt = f"""
You are an Industrial Data Science Mentor with deep expertise in machine learning, data engineering, and analytics.

Dataset Context:
{context_text}

User Question: {question}

Provide a comprehensive, professional response that:
1. Directly answers the question using available data
2. Provides actionable insights and recommendations
3. Explains technical concepts clearly
4. Suggests next steps or improvements
5. Uses industry best practices

Be specific with numbers, metrics, and concrete advice. Act as a senior data scientist mentoring a junior colleague.
"""

    response = llm(system_prompt)[0]['generated_text']

    # Clean up the response (remove the prompt if it gets included)
    if "User Question:" in response:
        response = response.split("User Question:")[1].strip()

    return response.strip()

def get_chat_response(dataset_id: str, question: str, history: List[Dict] = None):
    try:
        # Get comprehensive context
        context = retrieve_context(dataset_id, question)

        # Generate industrial-level response
        answer = generate_industrial_response(question, context, dataset_id)

        # Add some structured data if possible
        structured_data = None

        # Try to extract metrics or numbers from context for structured display
        if "accuracy" in question.lower() or "performance" in question.lower():
            # Look for model metrics in context
            for doc in context:
                if "metrics" in doc.lower():
                    try:
                        # Extract metrics if available
                        metrics_match = re.search(r"'metrics': ({[^}]+})", doc)
                        if metrics_match:
                            metrics_dict = eval(metrics_match.group(1))
                            structured_data = {
                                "type": "metrics",
                                "data": metrics_dict
                            }
                    except:
                        pass

        return {
            "status": "ok",
            "answer": answer,
            "context_used": context[:3],  # Return top 3 context items
            "structured_data": structured_data
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Industrial chat failed: {str(e)}. Please try rephrasing your question."
        }
