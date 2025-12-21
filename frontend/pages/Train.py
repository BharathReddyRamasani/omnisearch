# import streamlit as st
# import requests

# API = "http://127.0.0.1:8000"

# st.set_page_config(page_title="Train Model", layout="wide")
# st.title("🧠 Train Model")

# # ---------------- SESSION CHECK ----------------
# if "dataset_id" not in st.session_state:
#     st.warning("Upload dataset first")
#     st.stop()

# dataset_id = st.session_state["dataset_id"]

# # ---------------- LOAD SCHEMA ----------------
# schema_resp = requests.get(
#     f"{API}/schema",
#     params={"dataset_id": dataset_id}
# )

# if schema_resp.status_code != 200:
#     st.error("Schema not found. Upload dataset again.")
#     st.stop()

# schema = schema_resp.json()["schema"]

# st.subheader("Select Target Column")
# analysis = schema_resp.json()["target_analysis"]

# valid_targets = [
#     col for col, info in analysis.items() if info["valid"]
# ]

# invalid_targets = {
#     col: info["reasons"]
#     for col, info in analysis.items() if not info["valid"]
# }

# target = st.selectbox(
#     "Target",
#     valid_targets
# )

# with st.expander("🚫 Blocked targets (why you can’t select them)"):
#     for col, reasons in invalid_targets.items():
#         st.write(f"❌ **{col}** → {', '.join(reasons)}")


# # ---------------- TRAIN ----------------
# if st.button("🚀 Train Model", use_container_width=True):

#     with st.spinner("Training model..."):
#         resp = requests.post(
#             f"{API}/train",
#             params={
#                 "dataset_id": dataset_id,
#                 "target": target
#             }
#         )

#     if resp.status_code != 200:
#         st.error(resp.text)
#         st.stop()

#     result = resp.json()

#     if result.get("status") != "ok":
#         st.error(result)
#         st.stop()

#     st.success("Model trained successfully 🎉")

#     # ---------------- LOAD META (SOURCE OF TRUTH) ----------------
#     meta_resp = requests.get(
#         f"{API}/meta",
#         params={"dataset_id": dataset_id}
#     )

#     if meta_resp.status_code != 200:
#         st.error("Failed to load model metadata")
#         st.stop()

#     meta = meta_resp.json()

#     # ---------------- DISPLAY INFO ----------------
#     st.markdown("### 📌 Training Summary")
#     st.write("**Task Type:**", meta["task"])
#     st.write("**Best Model:**", meta.get("best_model", "N/A"))
#     st.write("**Model Version:**", meta.get("current_version", "N/A"))

#     st.markdown("### 📊 Evaluation Metrics")
#     if meta.get("metrics"):
#         st.json(meta["metrics"])
#     else:
#         st.info("No metrics available")

#     st.markdown("### ⭐ Top Influential Features")
#     st.write(meta.get("top_features", []))
import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Train Model", layout="wide")
st.title("🧠 Train Model")

# ---------------- SESSION CHECK ----------------
if "dataset_id" not in st.session_state:
    st.warning("Upload dataset first")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ---------------- LOAD SCHEMA ----------------
schema_resp = requests.get(
    f"{API}/schema",
    params={"dataset_id": dataset_id}
)

if schema_resp.status_code != 200:
    st.error("Schema not found. Upload dataset again.")
    st.stop()

schema = schema_resp.json()["schema"]
analysis = schema_resp.json()["target_analysis"]

valid_targets = [
    col for col, info in analysis.items() if info["valid"]
]

invalid_targets = {
    col: info["reasons"]
    for col, info in analysis.items() if not info["valid"]
}

st.subheader("Select Target Column")

target = st.selectbox(
    "Target Column",
    options=valid_targets,
    index=0 if valid_targets else None,
    help="Only columns that pass viability checks can be selected as target"
)

if not valid_targets:
    st.error("No valid target columns found in this dataset.")
    st.stop()

with st.expander("🚫 Blocked targets (why you can’t select them)", expanded=False):
    if invalid_targets:
        for col, reasons in invalid_targets.items():
            st.write(f"**{col}** → {', '.join(reasons)}")
    else:
        st.info("All columns passed validation checks")

# ---------------- TRAIN ----------------
if st.button("🚀 Train Model", type="primary", use_container_width=True):
    with st.spinner("Training model with 5-fold cross-validation..."):
        resp = requests.post(
            f"{API}/train",
            params={
                "dataset_id": dataset_id,
                "target": target
            }
        )

    if resp.status_code != 200:
        try:
            error_detail = resp.json()
            st.error(f"Training failed: {error_detail.get('detail', resp.text)}")
        except:
            st.error(f"Server error {resp.status_code}: {resp.text}")
        st.stop()

    result = resp.json()

    if result.get("status") != "ok":
        st.error(f"Training failed: {result.get('error', result)}")
        st.stop()

    st.success("Model trained successfully! 🎉")

    # ---------------- LOAD META (SOURCE OF TRUTH) ----------------
    meta_resp = requests.get(
        f"{API}/meta",
        params={"dataset_id": dataset_id}
    )

    if meta_resp.status_code != 200:
        st.error("Failed to load model metadata after training")
        st.stop()

    meta = meta_resp.json()

    # ---------------- DISPLAY TRAINING SUMMARY ----------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Task Type", meta["task"].title())
    with col2:
        st.metric("Selected Model", meta.get("best_model", "N/A"))
    with col3:
        st.metric("Reason for Model Choice", meta.get("model_reason", "Default selection"))

    # ---------------- CROSS-VALIDATION METRICS ----------------
    st.markdown("### 📊 Cross-Validation Results (5-fold)")
    cv_mean = meta["metrics"].get("cv_mean")
    cv_std = meta["metrics"].get("cv_std")
    cv_folds = meta["metrics"].get("cv_folds", 5)

    if cv_mean is not None:
        scoring = "Accuracy" if meta["task"] == "classification" else "R² Score"
        st.write(f"**Mean {scoring}:** {cv_mean:.4f}  ±  {cv_std:.4f} (std dev across {cv_folds} folds)")
        
        progress_value = cv_mean if meta["task"] == "classification" else max(0, min(1, (cv_mean + 1) / 2))
        st.progress(progress_value)

    # ---------------- HOLD-OUT TEST METRICS (SCALAR ONLY) ----------------
    st.markdown("### 📈 Hold-out Test Set Performance")

    task = meta["task"]
    metrics = meta["metrics"]

    if task == "classification":
        st.metric("Accuracy", round(metrics.get("accuracy", 0), 3))
        st.metric("Precision", round(metrics.get("precision", 0), 3))
        st.metric("Recall", round(metrics.get("recall", 0), 3))

    else:  # regression
        st.metric("RMSE", round(metrics.get("rmse", 0), 2))
        st.metric("MAE", round(metrics.get("mae", 0), 2))
        st.metric("R²", round(metrics.get("r2", 0), 3))

    # ---------------- CONFUSION MATRIX HEATMAP (ONLY FOR CLASSIFICATION) ----------------
    if task == "classification" and "confusion_matrix" in metrics:
        st.subheader("Confusion Matrix Heatmap")

        cm = np.array(metrics["confusion_matrix"])

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            linewidths=.5,
            ax=ax,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)

        # Optional: add class names if available in future
        st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ----------------
    if meta.get("top_features"):
        st.markdown("### ⭐ Top 6 Most Important Features")
        for i, feat in enumerate(meta["top_features"], 1):
            st.write(f"{i}. **{feat}**")

    st.balloons()