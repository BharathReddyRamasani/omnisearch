import streamlit as st
import requests

st.title("📤 **Upload Dataset**")

uploaded_file = st.file_uploader("📁 Drop CSV file", type="csv")

if uploaded_file:
    with st.spinner("Uploading..."):
        files = {'file': uploaded_file.getvalue()}
        resp = requests.post("http://127.0.0.1:8000/upload", files=files)
        
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.dataset_id = data["dataset_id"]
            st.success(f"✅ **{uploaded_file.name} loaded!** Dataset ID: `{data['dataset_id']}`")
            
            # SAFE PREVIEW
            if "columns" in data:
                st.json({
                    "Columns": len(data["columns"]),
                    "Preview": data.get("preview", [])[:2]
                })
            else:
                st.info("✅ Dataset ready! Go to **EDA** page")
        else:
            st.error("Upload failed!")
