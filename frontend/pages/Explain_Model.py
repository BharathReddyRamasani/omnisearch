import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("🤖 Model Reasoning")

if "dataset_id" not in st.session_state:
    st.stop()

meta = requests.get(
    f"{API}/meta",
    params={"dataset_id": st.session_state["dataset_id"]}
).json()

st.subheader("Why this model?")

reason = []

if meta["task"] == "classification":
    reason.append("Target has limited unique classes")

if meta.get("best_model"):
    reason.append(f"Selected model: {meta['best_model']}")

reason.append(
    "Model choice constrained based on data size and stability"
)

for r in reason:
    st.write("•", r)

st.subheader("What influences predictions?")
for f in meta.get("top_features", []):
    st.write("•", f)
