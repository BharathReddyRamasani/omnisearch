import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("📤 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV / Excel file",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    files = {
        "file": (uploaded_file.name, uploaded_file.getvalue())
    }

    resp = requests.post(f"{API}/upload", files=files)

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    data = resp.json()

    if data.get("status") != "ok":
        st.error(data)
        st.stop()

    # ---- SAVE STATE ----
    st.session_state["dataset_id"] = data["dataset_id"]

    st.success("Dataset uploaded successfully")

    st.write("**Dataset ID:**", data["dataset_id"])

    st.write("**Total Columns:**", len(data["columns"]))
    st.write("**Column Names:**")
    st.write(data["columns"])

    st.subheader("Preview (first 5 rows)")
    st.dataframe(data["preview"])
