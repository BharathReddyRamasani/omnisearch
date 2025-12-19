import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV / Excel",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    with st.spinner("Uploading..."):
        resp = requests.post(f"{API}/upload", files=files)

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    data = resp.json()

    # --- SESSION STATE ---
    st.session_state["dataset_id"] = data["dataset_id"]

    st.success("Upload successful")

    # --- SAFE ACCESS ---
    columns = data.get("columns", [])
    preview = data.get("preview", [])

    st.write("**Dataset ID:**", data["dataset_id"])
    st.write("**Total Columns:**", len(columns))

    if columns:
        st.write("**Columns:**")
        st.write(columns)

    if preview:
        st.subheader("Preview (Top 5 rows)")
        st.dataframe(preview)
