import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("🔮 Predict")

if "dataset_id" not in st.session_state:
    st.warning("Upload & train model first")
    st.stop()

# ---------------- FETCH SCHEMA ----------------
schema_resp = requests.get(
    f"{API}/schema",
    params={"dataset_id": st.session_state["dataset_id"]}
)

if schema_resp.status_code != 200:
    st.error("Schema not found. Upload dataset again.")
    st.stop()

schema_json = schema_resp.json()

if schema_json.get("status") != "ok":
    st.error(schema_json)
    st.stop()

schema = schema_json["schema"]  # ✅ THIS IS THE FIX

# ---------------- INPUT FORM ----------------
input_data = {}

st.subheader("Enter feature values")

for col, dtype in schema.items():

    # dtype is STRING now ("int64", "float64", "object")
    if dtype.startswith(("int", "float")):
        input_data[col] = st.number_input(col, value=0.0)
    else:
        input_data[col] = st.text_input(col)

# ---------------- PREDICT ----------------
if st.button("Predict"):
    resp = requests.post(
        f"{API}/predict",
        params={"dataset_id": st.session_state["dataset_id"]},
        json=input_data
    )

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    result = resp.json()

    if result.get("status") != "ok":
        st.error(result)
    else:
        st.success("Prediction Result")
        st.write(result["prediction"])
