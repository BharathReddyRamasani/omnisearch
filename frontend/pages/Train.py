import streamlit as st
import requests

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

st.subheader("Select Target Column")
target = st.selectbox("Target", list(schema.keys()))

# ---------------- TRAIN ----------------
if st.button("🚀 Train Model", use_container_width=True):

    with st.spinner("Training model..."):
        resp = requests.post(
            f"{API}/train",
            params={
                "dataset_id": dataset_id,
                "target": target
            }
        )

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    result = resp.json()

    if result.get("status") != "ok":
        st.error(result)
        st.stop()

    st.success("Model trained successfully 🎉")

    # ---------------- LOAD META (SOURCE OF TRUTH) ----------------
    meta_resp = requests.get(
        f"{API}/meta",
        params={"dataset_id": dataset_id}
    )

    if meta_resp.status_code != 200:
        st.error("Failed to load model metadata")
        st.stop()

    meta = meta_resp.json()

    # ---------------- DISPLAY INFO ----------------
    st.markdown("### 📌 Training Summary")
    st.write("**Task Type:**", meta["task"])
    st.write("**Best Model:**", meta.get("best_model", "N/A"))
    st.write("**Model Version:**", meta.get("current_version", "N/A"))

    st.markdown("### 📊 Evaluation Metrics")
    if meta.get("metrics"):
        st.json(meta["metrics"])
    else:
        st.info("No metrics available")

    st.markdown("### ⭐ Top Influential Features")
    st.write(meta.get("top_features", []))
