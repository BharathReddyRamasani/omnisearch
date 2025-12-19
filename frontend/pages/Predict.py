import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="Predict", layout="wide")
st.title("🔮 Prediction")

# ---------------- SESSION CHECK ----------------
if "dataset_id" not in st.session_state:
    st.warning("Upload & Train dataset first")
    st.stop()

dataset_id = st.session_state["dataset_id"]

# ---------------- META ----------------
meta = requests.get(
    f"{API}/meta",
    params={"dataset_id": dataset_id}
).json()

target = meta["target"]
task = meta["task"]
all_features = meta["features"]
top_features = meta["top_features"]

# ---------------- MODE ----------------
mode = st.radio(
    "Prediction Mode",
    ["Quick Predict (Recommended)", "Advanced Predict"],
    horizontal=True
)

features_to_use = top_features if mode.startswith("Quick") else all_features

# ---------------- SCHEMA (FIXED) ----------------
schema_resp = requests.get(
    f"{API}/schema",
    params={"dataset_id": dataset_id}
).json()

schema = schema_resp["schema"]   # ✅ FIX

# ---------------- INPUTS ----------------
input_data = {}

st.subheader("Provide Inputs")

for col in features_to_use:
    if col == target:
        continue

    if col not in schema:
        st.warning(f"Skipping `{col}` (not in schema)")
        continue

    dtype = schema[col]

    if dtype.startswith(("int", "float")):
        input_data[col] = st.number_input(col, value=0.0)
    else:
        input_data[col] = st.text_input(col, value="")

# ---------------- PREDICT ----------------
if st.button("🚀 Predict", use_container_width=True):
    resp = requests.post(
        f"{API}/predict",
        json={
            "dataset_id": dataset_id,
            "input_data": input_data
        }
    )

    result = resp.json()

    if "prediction" not in result:
        st.error(result)
        st.stop()

    st.success("Prediction Successful")

    if task == "classification":
        st.metric("Predicted Class", result["prediction"])
    else:
        st.metric("Predicted Value", round(float(result["prediction"]), 2))
