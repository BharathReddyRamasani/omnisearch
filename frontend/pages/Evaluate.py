import streamlit as st
import requests

API = "http://127.0.0.1:8003"

st.title("📊 Model Evaluation")

if "dataset_id" not in st.session_state:
    st.warning("Train a model first")
    st.stop()

meta = requests.get(
    f"{API}/meta",
    params={"dataset_id": st.session_state["dataset_id"]}
).json()

metrics = meta.get("metrics", {})

if not metrics:
    st.info("No metrics found. Train model again.")
    st.stop()

task = meta["task"]

if task == "classification":
    st.metric("Accuracy", round(metrics["accuracy"], 3))
    st.metric("Precision", round(metrics["precision"], 3))
    st.metric("Recall", round(metrics["recall"], 3))

    st.subheader("Confusion Matrix")
    st.dataframe(metrics["confusion_matrix"])

else:
    st.metric("RMSE", round(metrics["rmse"], 2))
    st.metric("MAE", round(metrics["mae"], 2))
    st.metric("R² Score", round(metrics["r2"], 3))
