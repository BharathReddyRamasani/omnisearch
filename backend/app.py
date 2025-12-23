# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uuid
# import os
# import pandas as pd
# import numpy as np
# from backend.services.ingest import process_upload, get_schema_logic, get_meta_logic
# from backend.services.eda import generate_eda
# from backend.services.training import train_model_logic
# from backend.services.predict import make_prediction
# from backend.services.chat import get_chat_response

# app = FastAPI(title="OmniSearch AI 🚀")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8501"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def safe_json_serializable(obj):
#     """Convert NaN/Inf to None for JSON"""
#     if isinstance(obj, float):
#         if np.isnan(obj) or np.isinf(obj):
#             return None
#     return obj

# @app.post("/upload")
# async def upload_csv(file: UploadFile = File(...)):
#     """Fast CSV upload WITH JSON-safe preview"""
#     dataset_id = str(uuid.uuid4())[:8]
    
#     # Create data dir
#     os.makedirs("data/datasets", exist_ok=True)
    
#     # FAST SAVE
#     filepath = f"data/datasets/{dataset_id}.csv"
#     with open(filepath, "wb") as f:
#         content = await file.read()
#         f.write(content)
    
#     # SAFE SCHEMA PREVIEW (JSON compliant)
#     try:
#         df_preview = pd.read_csv(filepath, nrows=5)
#         columns = list(df_preview.columns)
        
#         # JSON-SAFE PREVIEW (convert NaN to None)
#         preview = df_preview.head(2).replace({np.nan: None}).to_dict(orient='records')
#         row_count = len(pd.read_csv(filepath))  # Full count
        
#         return {
#             "status": "ok",
#             "dataset_id": dataset_id,
#             "columns": columns,
#             "preview": preview,
#             "rows": row_count
#         }
#     except Exception as e:
#         return {
#             "status": "ok",
#             "dataset_id": dataset_id,
#             "columns": [],
#             "preview": [],
#             "rows": 0,
#             "message": f"Preview failed: {str(e)}"
#         }

# @app.get("/schema/{dataset_id}")
# def get_schema(dataset_id: str):
#     try:
#         result = get_schema_logic(dataset_id)
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))

# @app.get("/meta/{dataset_id}")
# def get_meta(dataset_id: str):
#     try:
#         result = get_meta_logic(dataset_id)
#         return JSONResponse(content=result)
#     except Exception as e:
#         raise HTTPException(status_code=404, detail=str(e))

# @app.get("/eda/{dataset_id}")
# def run_eda(dataset_id: str):
#     try:
#         result = generate_eda(dataset_id)
#         return JSONResponse(content=result)
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.post("/train/{dataset_id}")
# def train_model(dataset_id: str, data: dict):
#     try:
#         result = train_model_logic(dataset_id, data.get("target", "auto"))
#         return JSONResponse(content=result)
#     except Exception as e:
#         return {"status": "failed", "error": str(e)}

# @app.post("/predict/{dataset_id}")
# def predict(dataset_id: str, input_data: dict):
#     try:
#         result = make_prediction(dataset_id, input_data)
#         return JSONResponse(content=result)
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.post("/chat/{dataset_id}")
# def chat_with_data(dataset_id: str, question: str):
#     try:
#         result = get_chat_response(dataset_id, question)
#         return JSONResponse(content=result)
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sklearn.metrics import r2_score, accuracy_score
import uuid
import os
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = FastAPI(title="OmniSearch AI 🚀")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    dataset_id = str(uuid.uuid4())[:8]
    os.makedirs("data", exist_ok=True)
    
    filepath = f"data/{dataset_id}.csv"
    with open(filepath, "wb") as f:
        f.write(await file.read())
    
    # SAFE PREVIEW
    try:
        df = pd.read_csv(filepath, nrows=5).replace({np.nan: None})
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "preview": df.head(2).to_dict('records'),
            "rows": len(pd.read_csv(filepath))
        }
    except:
        return {"status": "ok", "dataset_id": dataset_id, "columns": [], "preview": [], "rows": 0}

@app.get("/eda/{dataset_id}")
def run_eda(dataset_id: str):
    try:
        filepath = f"data/{dataset_id}.csv"
        if not os.path.exists(filepath):
            return {"status": "error", "message": "File not found"}
        
        df = pd.read_csv(filepath)
        
        # INDUSTRIAL QUALITY SCORE ALGORITHM
        total_rows = len(df)
        total_missing = df.isnull().sum().sum()
        missing_pct = total_missing / (total_rows * len(df.columns)) * 100
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        outlier_score = 0
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_score += outliers / total_rows * 100
        
        outlier_pct = outlier_score / len(numeric_cols) if len(numeric_cols) > 0 else 0
        
        # PROFESSIONAL QUALITY SCORE (0-100)
        quality_score = max(0, min(100, 100 - (missing_pct * 0.6 + outlier_pct * 0.4)))
        
        return {
            "status": "ok",
            "eda": {
                "rows": total_rows,
                "columns": len(df.columns),
                "missing": df.isnull().sum().to_dict(),
                "missing_total": int(total_missing),
                "missing_pct": round(missing_pct, 2),
                "outlier_pct": round(outlier_pct, 2),
                "quality_score": round(quality_score, 1),
                "quality_grade": "A" if quality_score >= 90 else "B" if quality_score >= 80 else "C",
                "recommendations": [
                    "✅ Production ready" if quality_score >= 90 else 
                    "⚠️ Minor cleaning" if quality_score >= 70 else 
                    "🚨 Heavy preprocessing needed"
                ],
                "summary": df.describe().round(2).to_dict()
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
@app.post("/train/{dataset_id}")
def train_model(dataset_id: str, data: dict = None):
    try:
        user_target = data.get("target") if data else None
        filepath = f"data/{dataset_id}.csv"
        df = pd.read_csv(filepath)
        
        # USER TARGET PRIORITY
        if user_target and user_target in df.columns:
            target_col = user_target
            print(f"✅ USER TARGET: {target_col}")
        else:
            # AUTO FALLBACK (previous smart logic)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            exclude_cols = ['Id', 'id', 'ID', 'index']
            numeric_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]
            
            # High variance first
            target_col = None
            for col in numeric_cols:
                if df[col].std() > 0.5:
                    target_col = col
                    break
            if target_col is None:
                target_col = numeric_cols[-1] if numeric_cols else None
        
        if target_col not in df.columns:
            return {"status": "error", "message": f"Target '{target_col}' not found"}
        
        # REST OF PIPELINE (same as before)
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].copy()
        y = df[target_col]
        
        # Safe features...
        safe_numeric = [col for col in df.select_dtypes(include=['number']).columns 
                       if col != target_col and col in X.columns]
        safe_categorical = [col for col in df.select_dtypes(include=['object']).columns 
                           if col in X.columns and df[col].nunique() < 20]
        
        transformers = []
        if safe_numeric:
            transformers.append(('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), safe_numeric))
        if safe_categorical:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), safe_categorical))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        # Model selection
        if y.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            task = "classification"
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            task = "regression"
        
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        score = accuracy_score(y_test, y_pred) if task == "classification" else r2_score(y_test, y_pred)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump({
            'pipeline': pipeline, 'target': target_col, 'task': task,
            'score': score, 'features': feature_cols
        }, f"models/{dataset_id}.pkl")
        
        return {
            "status": "ok", "target": target_col, "task": task,
            "score": round(float(score), 3),
            "features_used": len(safe_numeric) + len(safe_categorical),
            "message": f"✅ {task.title()} model trained on '{target_col}'! Score: {score:.3f}"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)[:100]}
    
#---------------------------------------------------------
@app.get("/meta/{dataset_id}")
def get_model_meta(dataset_id: str):
    """Check if model exists + return metadata"""
    try:
        model_path = f"models/{dataset_id}.pkl"
        if not os.path.exists(model_path):
            return {"status": "error", "message": "No trained model"}
        
        model_info = joblib.load(model_path)
        return {
            "status": "ok",
            "target": model_info.get('target'),
            "task": model_info.get('task'),
            "score": model_info.get('score'),
            "features": model_info.get('features', [])
        }
    except:
        return {"status": "error", "message": "Model metadata unavailable"}


#-------------------------------------------------------------
@app.post("/predict/{dataset_id}")
def predict(dataset_id: str, data: dict):
    try:
        model_path = f"models/{dataset_id}.pkl"
        if not os.path.exists(model_path):
            return {"status": "error", "message": "Model not trained"}
        
        model_info = joblib.load(model_path)
        pipeline = model_info['pipeline']
        
        # SAFE INPUT PROCESSING
        input_data = data.get("input_data", {})
        input_df = pd.DataFrame([input_data])
        
        # PREDICT
        prediction = pipeline.predict(input_df)[0]
        
        return {
            "status": "ok",
            "prediction": float(prediction),
            "target": model_info['target'],
            "task": model_info['task']
        }
    except Exception as e:
        return {"status": "error", "message": f"Prediction error: {str(e)[:100]}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
