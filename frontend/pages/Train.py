import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("🧠 Train Model")

if "dataset_id" not in st.session_state:
    st.warning("Upload dataset first")
    st.stop()

# ---------------- FETCH SCHEMA ----------------
resp = requests.get(
    f"{API}/schema",
    params={"dataset_id": st.session_state["dataset_id"]}
)

if resp.status_code != 200:
    st.error("Failed to fetch schema")
    st.stop()

data = resp.json()

if data.get("status") != "ok":
    st.error(data)
    st.stop()

schema = data["schema"]
columns = list(schema.keys())

# ---------------- TARGET SELECTION ----------------
target = st.selectbox("Select Target Column", columns)

# ---------------- TRAIN ----------------
if st.button("Train Model"):
    resp = requests.post(
        f"{API}/train",
        params={
            "dataset_id": st.session_state["dataset_id"],
            "target": target
        }
    )

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    result = resp.json()

    if result.get("status") == "ok":
        st.success("Model trained successfully")
        st.write("Task Type:", result["task"])
    else:
        st.error(result)
