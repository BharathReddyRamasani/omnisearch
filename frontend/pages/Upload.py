import streamlit as st
import requests

API = "http://127.0.0.1:8000/api"

st.header("📤 Upload Dataset")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    with st.spinner("Uploading..."):
        res = requests.post(
            f"{API}/upload",
            files={"file": (file.name, file.getvalue())}
        )

    if res.status_code == 200:
        data = res.json()
        st.session_state.dataset_id = data["dataset_id"]

        st.success(f"Uploaded successfully! Dataset ID: {data['dataset_id']}")
        st.write("Columns:", data["columns"])
        st.write("Preview:")
        st.dataframe(data["preview"])
    else:
        st.error("Upload failed")
