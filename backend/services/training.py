import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           confusion_matrix, mean_squared_error, mean_absolute_error, r2_score)
from backend.services.utils import load_df, clean_dataframe, dataset_dir, validate_target, safe_json
from backend.services.utils import load_df, clean_dataframe, dataset_dir, safe_json


def train_model_logic(dataset_id: str, target: str = "auto"):
    try:
        df = clean_dataframe(load_df(dataset_id))
        
        # AUTO-DETECT TARGET
        if target == "auto":
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                target = numeric_cols[-1]  # Last numeric column
            else:
                return {"status": "failed", "error": "No suitable target column"}
        
        if target not in df.columns:
            return {"status": "failed", "error": "Target column not found"}
        
        reason = validate_target(df[target])
        if reason:
            return {"status": "failed", "error": reason}
        
        df = df.dropna(subset=[target])
        if df.shape[0] < 20:  # Reduced minimum
            return {"status": "failed", "error": "Not enough data (<20 rows)"}
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Drop useless columns
        DROP_COLS = ['alley', 'poolqc', 'fence', 'miscfeature', 'fireplacequ', 'garageyrblt']
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors='ignore')
        
        num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = [c for c in X.select_dtypes(include=['object']).columns if X[c].nunique() < 20]
        
        features = num_cols + cat_cols
        if not features:
            return {"status": "failed", "error": "No usable features"}
        
        # FAST PREPROCESSOR
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ]), cat_cols)
            ]
        )
        
        # Model selection (FAST)
        if y.dtype == 'object' or y.nunique() <= 10:
            task = "classification"
            model = RandomForestClassifier(n_estimators=50, random_state=42)  # FAST
            model_reason = "Random Forest (Classification)"
        else:
            task = "regression"
            model = RandomForestRegressor(n_estimators=50, random_state=42)  # FAST
            model_reason = "Random Forest (Regression)"
        
        pipeline = Pipeline([('prep', preprocessor), ('model', model)])
        
        # FAST TRAIN/TEST (no stratify for speed)
        X_train, X_test, y_train, y_test = train_test_split(
            X[features], y, test_size=0.2, random_state=42
        )
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        
        # Metrics (SIMPLE)
        if task == "classification":
            test_metrics = {
                "accuracy": float(accuracy_score(y_test, preds))
            }
        else:
            test_metrics = {
                "r2": float(r2_score(y_test, preds))
            }
        
        # FAST CV (3-fold instead of 5)
        cv_scores = cross_val_score(pipeline, X[features], y, cv=3, n_jobs=-1)
        
        metrics = {
            "test_metrics": test_metrics,
            "cv_scores": cv_scores.tolist(),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std())
        }
        
        # Feature importance (SIMPLE)
        model_step = pipeline.named_steps['model']
        top_features = []
        if hasattr(model_step, 'feature_importances_'):
            importances = model_step.feature_importances_
            ranked = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            top_features = [f.split('_')[0] for f, _ in ranked[:5]]
        
        # Save model
        dpath = dataset_dir(dataset_id)
        joblib.dump(pipeline, os.path.join(dpath, "model.pkl"))
        
        meta = {
            "target": target,
            "task": task,
            "features": features,
            "top_features": top_features,
            "best_model": model.__class__.__name__,
            "model_reason": model_reason,
            "metrics": metrics
        }
        
        with open(os.path.join(dpath, "meta.json"), 'w') as f:
            json.dump(meta, f, indent=2, default=safe_json)
        
        return {
            "status": "ok",
            "task": task,
            "accuracy": metrics["cv_mean"],
            "top_features": top_features[:3]
        }
    except Exception as e:
        return {"status": "failed", "error": str(e)}
