"""
PRODUCTION PREDICTION WITH:
✅ Confidence/probability outputs
✅ Input schema validation
✅ Friendly error messages
✅ Auto-fill defaults
✅ Feature mapping to normalized names
"""
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from backend.services.utils import model_dir, load_meta


def validate_input_schema(payload: Dict, required_features: List[str], 
                         feature_defaults: Dict) -> tuple[bool, Optional[str]]:
    """
    Validate input payload against expected schema.
    
    Returns:
        (is_valid, error_message)
    """
    if not isinstance(payload, dict):
        return False, "Input must be a JSON object"
    
    # Check for required fields
    provided_keys = set(payload.keys())
    required_keys = set(required_features)
    
    missing_keys = required_keys - provided_keys
    if missing_keys:
        return False, f"Missing required fields: {', '.join(missing_keys)}"
    
    # Validate types
    for key, value in payload.items():
        if key not in feature_defaults:
            return False, f"Unknown feature: '{key}'"
        
        default_val = feature_defaults[key]
        
        # Check type compatibility
        if isinstance(default_val, (int, float)):
            if value is not None and not isinstance(value, (int, float, str)):
                return False, f"Field '{key}' must be numeric, got {type(value).__name__}"
            
            # Try to convert to numeric
            if isinstance(value, str):
                try:
                    float(value)
                except ValueError:
                    return False, f"Field '{key}' cannot be converted to numeric"
        else:
            # Categorical field
            if value is not None and not isinstance(value, str):
                return False, f"Field '{key}' must be text, got {type(value).__name__}"
    
    return True, None


def autofill_missing_features(payload: Dict, feature_defaults: Dict, 
                              top_features: List[str] = None) -> Dict:
    """
    Auto-fill missing features with defaults.
    Prioritizes provided values, falls back to defaults.
    """
    filled = {}
    
    # Determine which features to fill
    features_to_fill = top_features if top_features else list(feature_defaults.keys())
    
    for feature in features_to_fill:
        if feature in payload and payload[feature] not in [None, ""]:
            filled[feature] = payload[feature]
        elif feature in feature_defaults:
            filled[feature] = feature_defaults[feature]
        else:
            # Safe fallback
            filled[feature] = 0 if isinstance(feature_defaults.get(feature), (int, float)) else "UNKNOWN"
    
    return filled


def make_prediction(dataset_id: str, payload: Dict, mode: str = "full") -> Dict:
    """
    Make predictions with confidence scores and input validation.
    
    Args:
        dataset_id: Dataset identifier
        payload: Input features
        mode: "full" (all features) or "smart" (top features)
    
    Returns:
        {
            "status": "ok" | "failed",
            "prediction": value or error message,
            "confidence": float or None,
            "confidence_type": "probability" | "softmax_max" | "none",
            "used_features": list of features used,
            "auto_filled": list of auto-filled features,
            "mode": prediction mode used,
            "target": target variable name,
            "task": "classification" | "regression"
        }
    """
    try:
        # Load model
        model_path = os.path.join(model_dir(dataset_id), "model.pkl")
        if not os.path.exists(model_path):
            return {
                "status": "failed",
                "error": "Model not found - train a model first",
                "error_code": "MODEL_NOT_FOUND"
            }
        
        model = joblib.load(model_path)
        
        # Load metadata
        try:
            meta = load_meta(dataset_id)
        except:
            return {
                "status": "failed",
                "error": "Model metadata not found",
                "error_code": "METADATA_NOT_FOUND"
            }
        
        # Extract metadata
        feature_defaults = meta.get("feature_defaults", {})
        top_features = meta.get("top_features", [])
        raw_columns = meta.get("raw_columns", [])
        target = meta.get("target")
        task = meta.get("task")
        feature_importance = meta.get("feature_importance", {})
        
        if not feature_defaults:
            return {
                "status": "failed",
                "error": "No feature defaults in model metadata",
                "error_code": "INVALID_METADATA"
            }
        
        # Determine features to use
        if mode == "smart" and top_features:
            required_features = top_features
        else:
            required_features = list(feature_defaults.keys())
        
        # Validate input
        is_valid, error_msg = validate_input_schema(payload, required_features, feature_defaults)
        if not is_valid:
            return {
                "status": "failed",
                "error": error_msg,
                "error_code": "INVALID_INPUT_SCHEMA",
                "hint": f"Expected features: {', '.join(required_features)}"
            }
        
        # Auto-fill missing features
        filled_payload = autofill_missing_features(payload, feature_defaults, required_features)
        auto_filled = [f for f in required_features if f not in payload or payload[f] in [None, ""]]
        
        # Prepare input DataFrame
        X = pd.DataFrame([filled_payload])
        
        # Make prediction
        try:
            pred = model.predict(X)[0]
        except Exception as e:
            return {
                "status": "failed",
                "error": f"Prediction failed: {str(e)}",
                "error_code": "PREDICTION_FAILED"
            }
        
        # Extract confidence
        confidence = None
        confidence_type = "none"
        
        try:
            # Try to get probability for classification
            if task == "classification" and hasattr(model.named_steps.get("model"), "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = float(np.max(proba))
                confidence_type = "probability"
            elif task == "classification" and hasattr(model.named_steps.get("model"), "decision_function"):
                # SVM case
                confidence = float(np.abs(model.decision_function(X)[0]))
                confidence_type = "decision_score"
        except:
            confidence = None
            confidence_type = "none"
        
        # ✅ FIX #3: APPLY DRIFT CONFIDENCE ADJUSTMENT
        drift_adjustment = meta.get("drift_confidence_adjustment", 1.0)
        drift_warning = None
        
        if drift_adjustment < 1.0 and confidence is not None:
            confidence = confidence * drift_adjustment
            drift_warning = (
                f"Statistical drift detected in training data. "
                f"Confidence reduced by {int((1-drift_adjustment)*100)}%. "
                f"Consider retraining on newer data."
            )
        
        # Build response
        return {
            "status": "ok",
            "prediction": float(pred) if task == "regression" else str(pred),
            "confidence": round(confidence, 4) if confidence else None,
            "confidence_type": confidence_type,
            "drift_warning": drift_warning,  # ✅ FIX #3: Include drift warning
            "used_features": list(filled_payload.keys()),
            "auto_filled": auto_filled,
            "auto_filled_count": len(auto_filled),
            "mode": mode,
            "target": target,
            "task": task,
            "feature_importance_top_5": dict(list(feature_importance.get("scores", {}).items())[:5]) if isinstance(feature_importance, dict) else {},
            "note": "Some features were auto-filled with training defaults" if auto_filled else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "failed",
            "error": f"Unexpected error: {str(e)}",
            "error_code": "INTERNAL_ERROR"
        }


def batch_predict(dataset_id: str, payloads: List[Dict], mode: str = "full") -> Dict:
    """
    Make batch predictions on multiple samples.
    
    Returns:
        {
            "status": "ok" | "failed",
            "predictions": [
                { prediction result for each sample }
            ],
            "successful_count": int,
            "failed_count": int
        }
    """
    try:
        if not isinstance(payloads, list):
            return {
                "status": "failed",
                "error": "Input must be a list of prediction payloads",
                "error_code": "INVALID_INPUT"
            }
        
        if len(payloads) > 10000:
            return {
                "status": "failed",
                "error": "Batch size too large (max 10000)",
                "error_code": "BATCH_SIZE_EXCEEDED"
            }
        
        predictions = []
        successful = 0
        failed = 0
        
        for i, payload in enumerate(payloads):
            result = make_prediction(dataset_id, payload, mode)
            result["batch_index"] = i
            predictions.append(result)
            
            if result.get("status") == "ok":
                successful += 1
            else:
                failed += 1
        
        return {
            "status": "ok" if failed == 0 else "partial",
            "predictions": predictions,
            "successful_count": successful,
            "failed_count": failed,
            "total_count": len(payloads)
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "BATCH_FAILED"
        }


def explain_prediction(dataset_id: str, payload: Dict) -> Dict:
    """
    Explain a prediction using feature importance.
    
    Returns:
        {
            "prediction": ...,
            "confidence": ...,
            "feature_importance": {
                "feature_name": {
                    "importance_score": float,
                    "used_value": actual value used,
                    "impact": "high" | "medium" | "low"
                }
            },
            "key_drivers": [list of top 3 features]
        }
    """
    try:
        # Get prediction
        pred_result = make_prediction(dataset_id, payload, mode="full")
        
        if pred_result.get("status") != "ok":
            return pred_result
        
        # Load metadata
        meta = load_meta(dataset_id)
        feature_importance = meta.get("feature_importance", {})
        
        if not isinstance(feature_importance, dict):
            feature_importance = {}
        
        scores = feature_importance.get("scores", {})
        
        # Rank features by importance
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Map to actual values used
        filled_payload = autofill_missing_features(payload, meta.get("feature_defaults", {}))
        
        feature_impacts = {}
        for feature, score in ranked:
            impact = "high" if score > 0.2 else "medium" if score > 0.05 else "low"
            feature_impacts[feature] = {
                "importance_score": round(float(score), 6),
                "used_value": filled_payload.get(feature),
                "impact": impact
            }
        
        return {
            "status": "ok",
            "prediction": pred_result.get("prediction"),
            "confidence": pred_result.get("confidence"),
            "target": meta.get("target"),
            "task": meta.get("task"),
            "feature_importance": feature_impacts,
            "key_drivers": [f for f, _ in ranked[:3]],
            "model": meta.get("best_model"),
            "model_score": meta.get("best_score")
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "EXPLANATION_FAILED"
        }
