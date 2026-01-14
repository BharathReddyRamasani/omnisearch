# # import os, joblib
# # import pandas as pd
# # from backend.services.utils import model_dir, load_meta



# # def make_prediction(dataset_id: str, payload: dict):
# #     root = model_dir(dataset_id)
# #     model_path = os.path.join(root, "model.pkl")

# #     if not os.path.exists(model_path):
# #         return {"status": "failed", "error": "Model not trained"}

# #     meta = load_meta(dataset_id)
# #     model = joblib.load(model_path)

# #     mode = payload.pop("_mode", "full")
# #     defaults = meta["feature_defaults"]
# #     top_features = meta.get("top_features", [])

# #     used = top_features if mode == "top" else defaults.keys()

# #     filled, auto = {}, []
# #     for f in used:
# #         if f in payload and payload[f] not in ["", None]:
# #             filled[f] = payload[f]
# #         else:
# #             filled[f] = defaults[f]
# #             auto.append(f)

# #     X = pd.DataFrame([filled])
# #     pred = model.predict(X)[0]

# #     conf = None
# #     if hasattr(model.named_steps["model"], "predict_proba"):
# #         conf = float(model.predict_proba(X).max())

# #     return {
# #         "status": "ok",
# #         "prediction": pred,
# #         "confidence": conf,
# #         "mode": mode,
# #         "used_features": list(used),
# #         "auto_filled": auto,
# #     }



# # import streamlit as st
# # import requests
# # import pandas as pd
# # import json

# # API = "http://127.0.0.1:8000/api"

# # st.set_page_config(
# #     page_title="OmniSearch AI – Enterprise Predict",
# #     layout="wide"
# # )

# # # =====================================================
# # # DATASET CHECK
# # # =====================================================
# # if "dataset_id" not in st.session_state:
# #     st.error("🚫 No dataset loaded. Upload & train a model first.")
# #     st.stop()

# # dataset_id = st.session_state.dataset_id

# # # =====================================================
# # # FETCH MODEL METADATA
# # # =====================================================
# # @st.cache_data(ttl=60)
# # def fetch_model_meta(ds):
# #     r = requests.get(f"{API}/meta/{ds}", timeout=15)
# #     if r.status_code != 200:
# #         return None
# #     return r.json()

# # model_meta = fetch_model_meta(dataset_id)

# # if not model_meta or model_meta.get("status") != "ok":
# #     st.error("🚫 No trained model found. Train a model first.")
# #     st.stop()

# # # -----------------------------------------------------
# # # SAFE METADATA ACCESS (CRITICAL FIX)
# # # -----------------------------------------------------
# # TOP_FEATURES = model_meta.get("top_features", [])
# # FEATURE_DEFAULTS = model_meta.get("feature_defaults", {})
# # RAW_COLUMNS = model_meta.get("raw_columns", [])
# # TARGET = model_meta.get("target")
# # TASK = model_meta.get("task")
# # BEST_MODEL = model_meta.get("best_model")
# # BEST_SCORE = model_meta.get("best_score")

# # # =====================================================
# # # HEADER
# # # =====================================================
# # st.markdown(
# #     f"""
# # <div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
# # padding:2rem;border-radius:16px;color:white;">
# # <h1>🔮 Enterprise Prediction Engine</h1>
# # <p>
# # Model: <b>{BEST_MODEL}</b> &nbsp; | &nbsp;
# # Task: <b>{TASK.upper()}</b> &nbsp; | &nbsp;
# # Score: <b>{BEST_SCORE}</b>
# # </p>
# # </div>
# # """,
# #     unsafe_allow_html=True
# # )

# # st.markdown("---")

# # # =====================================================
# # # MODE SELECTION
# # # =====================================================
# # mode = st.radio(
# #     "Prediction Mode",
# #     [
# #         "🧠 Smart Mode (Top Impact Features)",
# #         "🧾 Full Input Mode (All Features)"
# #     ],
# #     horizontal=True
# # )

# # # =====================================================
# # # DETERMINE FEATURES TO ASK USER
# # # =====================================================
# # if mode.startswith("🧠") and TOP_FEATURES:
# #     input_features = TOP_FEATURES
# #     st.info(
# #         "Smart Mode active: Only high-impact features required. "
# #         "Remaining fields are auto-filled using training-time defaults."
# #     )
# # else:
# #     input_features = RAW_COLUMNS
# #     st.info("Full Input Mode active: Provide all features.")

# # # =====================================================
# # # INPUT FORM
# # # =====================================================
# # st.markdown("## 📝 Input Features")

# # inputs = {}
# # missing_notice = False

# # with st.form("predict_form"):
# #     cols = st.columns(3)

# #     for i, feature in enumerate(input_features):
# #         default = FEATURE_DEFAULTS.get(feature)

# #         col = cols[i % 3]

# #         with col:
# #             # Numeric
# #             if isinstance(default, (int, float)):
# #                 inputs[feature] = st.number_input(
# #                     label=feature,
# #                     value=float(default),
# #                 )

# #             # Categorical
# #             else:
# #                 # Allow user to override default
# #                 inputs[feature] = st.text_input(
# #                     label=feature,
# #                     value=str(default) if default is not None else ""
# #                 )

# #             if default is None:
# #                 missing_notice = True

# #     submitted = st.form_submit_button("🚀 Predict", use_container_width=True)

# # # =====================================================
# # # WARN IF AUTO-FILL USED
# # # =====================================================
# # if missing_notice:
# #     st.caption(
# #         "ℹ️ Some fields were auto-filled using training-time defaults "
# #         "(median / most frequent)."
# #     )

# # # =====================================================
# # # PREDICTION
# # # =====================================================
# # if submitted:
# #     with st.spinner("🔍 Generating prediction..."):
# #         # ------------------------------------------------
# #         # AUTO-FILL REMAINING FEATURES (INDUSTRIAL FIX)
# #         # ------------------------------------------------
# #         final_payload = FEATURE_DEFAULTS.copy()
# #         final_payload.update(inputs)

# #         try:
# #             resp = requests.post(
# #                 f"{API}/predict/{dataset_id}",
# #                 json=final_payload,
# #                 timeout=20
# #             )

# #             if resp.status_code != 200:
# #                 st.error("Prediction failed")
# #                 st.json(resp.text)
# #                 st.stop()

# #             result = resp.json()
# #             if result.get("status") != "ok":
# #                 st.error(result.get("error", "Prediction failed"))
# #                 st.stop()

# #         except Exception as e:
# #             st.error(f"Connection error: {e}")
# #             st.stop()

# #     # =================================================
# #     # RESULT DISPLAY
# #     # =================================================
# #     st.markdown("---")
# #     st.markdown("## ✅ Prediction Result")

# #     c1, c2 = st.columns(2)

# #     with c1:
# #         st.metric(
# #             label=f"Predicted {TARGET}",
# #             value=str(result["prediction"])
# #         )

# #     with c2:
# #         if result.get("confidence") is not None:
# #             st.metric(
# #                 label="Confidence",
# #                 value=f"{result['confidence']*100:.1f}%"
# #             )
# #         else:
# #             st.metric(
# #                 label="Confidence",
# #                 value="N/A"
# #             )

# #     st.success("Prediction completed successfully")

# # # =====================================================
# # # EXPLAINABILITY SECTION (READY)
# # # =====================================================
# # if TOP_FEATURES:
# #     st.markdown("---")
# #     st.markdown("## 🔥 Key Drivers Used")
# #     st.write(", ".join(TOP_FEATURES))

# # st.caption("Enterprise Predict • Schema-Safe • Production-Ready")

# # backend/services/predict.py
# # backend/services/predict.py

# import os
# import joblib
# import pandas as pd
# import numpy as np
# from fastapi import HTTPException
# from backend.services.utils import model_dir, load_meta


# def _ai_autofill(feature, value, defaults):
#     """
#     Enterprise autofill logic
#     """
#     if value is not None and value != "":
#         return value

#     # 1️⃣ Training-time default
#     if feature in defaults:
#         return defaults[feature]

#     # 2️⃣ Numeric-safe fallback
#     if isinstance(defaults.get(feature), (int, float)):
#         return 0.0

#     # 3️⃣ Categorical-safe fallback
#     return "UNKNOWN"


# def make_prediction(dataset_id: str, payload: dict):
#     # -------------------------------------------------
#     # Load model
#     # -------------------------------------------------
#     model_path = os.path.join(model_dir(dataset_id), "model.pkl")
#     if not os.path.exists(model_path):
#         raise HTTPException(404, "Model not trained")

#     model = joblib.load(model_path)

#     # -------------------------------------------------
#     # Load metadata
#     # -------------------------------------------------
#     meta = load_meta(dataset_id)

#     raw_columns = meta.get("raw_columns")
#     feature_defaults = meta.get("feature_defaults", {})

#     if not raw_columns:
#         raise HTTPException(500, "Model schema missing raw_columns")

#     # -------------------------------------------------
#     # AI-Guided Auto Fill (CRITICAL FIX)
#     # -------------------------------------------------
#     row = {}
#     for col in raw_columns:
#         user_value = payload.get(col)
#         row[col] = _ai_autofill(col, user_value, feature_defaults)

#     X = pd.DataFrame([row])

#     # -------------------------------------------------
#     # Prediction
#     # -------------------------------------------------
#     try:
#         prediction = model.predict(X)[0]
#     except Exception as e:
#         raise HTTPException(
#             400,
#             f"Prediction failed due to invalid inputs: {str(e)}"
#         )

#     confidence = None
#     if hasattr(model.named_steps["model"], "predict_proba"):
#         proba = model.predict_proba(X)
#         confidence = float(np.max(proba))

#     return {
#         "status": "ok",
#         "prediction": prediction,
#         "confidence": confidence,
#         "autofilled_fields": [
#             k for k in raw_columns if k not in payload
#         ],
#     }

import os
import joblib
import pandas as pd
import numpy as np
import hashlib
import json
from fastapi import HTTPException

from backend.services.utils import model_dir, load_meta, load_raw, load_clean
from typing import Dict


# =====================================================
# DATASET HASH (MUST MATCH training.py)
# =====================================================
def dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df[sorted(df.columns)]
    return hashlib.md5(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()


# =====================================================
# AUTOFILL HELPER
# =====================================================
def _autofill(col, user_value, defaults):
    """Auto-fill missing values with defaults"""
    # Handle None, empty strings, NaN
    if user_value is None:
        return defaults.get(col, 0.0)
    if isinstance(user_value, str) and not user_value.strip():
        return defaults.get(col, 0.0)
    if isinstance(user_value, float) and pd.isna(user_value):
        return defaults.get(col, 0.0)
    
    # Return the value as-is if provided
    return user_value


# =====================================================
# MAIN PREDICTION LOGIC
# =====================================================
def make_prediction(dataset_id: str, payload: dict):
    from backend.services.model_registry import ModelRegistry
    from backend.services.training import detect_dataset_drift
    
    # Validate and clean dataset_id
    if not dataset_id or not isinstance(dataset_id, str):
        raise HTTPException(status_code=400, detail="Invalid dataset_id")
    
    dataset_id = str(dataset_id).strip()
    # Handle case where dataset_id has commas (malformed)
    if "," in dataset_id:
        dataset_id = dataset_id.split(",")[0].strip()
    
    # Get active model info from registry
    active_version = ModelRegistry.get_active_version(dataset_id)
    if not active_version:
        raise HTTPException(status_code=404, detail="No active model found for this dataset")

    model_path = active_version["model_path"]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    meta = active_version["metadata"]
    model = joblib.load(model_path)

    # -------------------------------------------------
    # VALIDATE DATASET CONSISTENCY (ENTERPRISE SAFETY)
    # -------------------------------------------------
    try:
        df_current = load_raw(dataset_id)
    except Exception:
        raise HTTPException(404, "Cannot load current dataset")

    # Use the new drift detection
    drift_check = detect_dataset_drift(dataset_id, df_current)
    if drift_check["drift_detected"]:
        # Return error response with drift details
        return {
            "status": "error",
            "error": "Dataset has changed since model training",
            "drift_details": drift_check,
            "recommendation": "Please retrain the model with the current dataset"
        }

    # -------------------------------------------------
    # FEATURE HANDLING
    # -------------------------------------------------
    raw_columns = [
        c for c in meta["raw_columns"]
        if c not in meta.get("id_columns", [])
    ]

    defaults = meta["feature_defaults"]
    top_features = meta.get("top_features", raw_columns[:8])  # safety

    # Rename mode: "top" → "guided", "full" → "full"
    internal_mode = payload.pop("_mode", "full")
    mode_label = "guided" if internal_mode == "top" else "full"

    used_features = top_features if internal_mode == "top" else raw_columns

    row = {}
    auto_filled = []

    for col in raw_columns:
        val = payload.get(col)
        # Clean value: handle None, empty strings, NaN
        if val is None or (isinstance(val, str) and not val.strip()):
            val = None
        elif isinstance(val, str):
            val = val.strip()
        
        final = _autofill(col, val, defaults)
        row[col] = final
        if val is None or (isinstance(val, str) and not val.strip()):
            auto_filled.append(col)

    X = pd.DataFrame([row])

    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------
    try:
        pred = model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    confidence = None
    model_step = model.named_steps.get("model")
    if hasattr(model_step, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))

    # -------------------------------------------------
    # RESPONSE (TRANSPARENT)
    # -------------------------------------------------
    return {
        "status": "ok",
        "prediction": pred,
        "confidence": confidence,
        "mode": mode_label,           # ← external: "guided" or "full"
        "used_features": used_features,
        "auto_filled": auto_filled,
    }


# =====================================================
# BATCH PREDICTION
# =====================================================
def make_batch_prediction(dataset_id: str, csv_content: bytes) -> Dict:
    """Make predictions on a batch of data from CSV"""
    from backend.services.model_registry import ModelRegistry
    from backend.services.training import detect_dataset_drift
    import io
    
    # Validate and clean dataset_id
    if not dataset_id or not isinstance(dataset_id, str):
        raise HTTPException(status_code=400, detail="Invalid dataset_id")
    
    dataset_id = str(dataset_id).strip()
    if "," in dataset_id:
        dataset_id = dataset_id.split(",")[0].strip()
    
    # Get active model info from registry
    active_version = ModelRegistry.get_active_version(dataset_id)
    if not active_version:
        raise HTTPException(status_code=404, detail="No active model found for this dataset")

    model_path = active_version["model_path"]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found")

    meta = active_version["metadata"]
    model = joblib.load(model_path)

    # Parse input CSV
    try:
        input_df = pd.read_csv(io.BytesIO(csv_content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if input_df.empty:
        raise HTTPException(status_code=400, detail="Input CSV is empty")

    # Check for dataset drift (using the input data as reference)
    try:
        current_df = load_raw(dataset_id)
        drift_check = detect_dataset_drift(dataset_id, current_df)
        if drift_check["drift_detected"]:
            error_details = {
                "error": "Dataset has changed since model training",
                "drift_details": drift_check,
                "recommendation": "Please retrain the model with the current dataset"
            }
            raise HTTPException(status_code=409, detail=json.dumps(error_details))
    except Exception:
        # If we can't check drift, continue but log warning
        pass

    # Process each row for prediction
    predictions = []
    errors = []
    
    raw_columns = [
        c for c in meta["raw_columns"]
        if c not in meta.get("dropped_id_columns", [])
    ]
    
    defaults = meta["feature_defaults"]
    
    for idx, row in input_df.iterrows():
        try:
            # Prepare features
            filled_row = {}
            auto_filled = []
            
            for col in raw_columns:
                val = row.get(col)
                if pd.isna(val) or val == "":
                    filled_row[col] = defaults.get(col, 0.0 if pd.api.types.is_numeric_dtype(type(defaults.get(col))) else "UNKNOWN")
                    auto_filled.append(col)
                else:
                    filled_row[col] = val
            
            X = pd.DataFrame([filled_row])
            
            # Make prediction
            pred = model.predict(X)[0]
            
            confidence = None
            model_step = model.named_steps.get("model")
            if hasattr(model_step, "predict_proba"):
                probs = model.predict_proba(X)[0]
                confidence = float(np.max(probs))
            
            predictions.append({
                "row_index": int(idx),
                "prediction": pred,
                "confidence": confidence,
                "auto_filled": auto_filled
            })
            
        except Exception as e:
            errors.append({
                "row_index": int(idx),
                "error": str(e)
            })
    
    # Create output CSV
    output_df = input_df.copy()
    output_df["prediction"] = [p["prediction"] for p in predictions] + [None] * len(errors)
    output_df["confidence"] = [p["confidence"] for p in predictions] + [None] * len(errors)
    
    # Convert to CSV string
    output_csv = output_df.to_csv(index=False)
    
    return {
        "status": "ok",
        "total_rows": len(input_df),
        "successful_predictions": len(predictions),
        "errors": len(errors),
        "predictions": predictions,
        "error_details": errors,
        "output_csv": output_csv
    }