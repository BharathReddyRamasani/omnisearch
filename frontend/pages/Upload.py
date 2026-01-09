# import streamlit as st
# import requests

# API = "http://127.0.0.1:8000/api"

# st.header("📤 Upload Dataset")

# file = st.file_uploader("Upload CSV file", type=["csv"])

# if file:
#     with st.spinner("Uploading..."):
#         res = requests.post(
#             f"{API}/upload",
#             files={"file": (file.name, file.getvalue())}
#         )

#     if res.status_code == 200:
#         data = res.json()
#         st.session_state.dataset_id = data["dataset_id"]

#         st.success(f"Uploaded successfully! Dataset ID: {data['dataset_id']}")
#         st.write("Columns:", data["columns"])
#         st.write("Preview:")
#         st.dataframe(data["preview"])
#     else:
#         st.error("Upload failed")

import streamlit as st
import requests

API = "http://127.0.0.1:8003/api"

st.set_page_config(
    page_title="OmniSearch AI – Upload",
    layout="wide"
)

st.markdown(
    """
    <div style="padding:2rem;border-radius:12px;
                background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
                color:white">
        <h1>📤 Dataset Upload</h1>
        <p>Secure ingestion • Schema validation • Session persistence</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"],
    help="Only CSV files are supported"
)

if uploaded_file is not None:
    with st.spinner("Uploading dataset…"):
        resp = requests.post(
            f"{API}/upload",
            files={"file": uploaded_file}
        )

    if resp.status_code != 200:
        st.error("Upload failed")
        st.code(resp.text)
        st.stop()

    data = resp.json()

    # ------------------ HARD GUARANTEE ------------------
    if "dataset_id" not in data or "columns" not in data:
        st.error("Backend response invalid")
        st.json(data)
        st.stop()

    # ------------------ SESSION STATE ------------------
    st.session_state.dataset_id = data["dataset_id"]
    st.session_state.columns = data["columns"]

    # Clear downstream caches safely
    st.session_state.pop("model_meta", None)
    st.session_state.pop("eda_data", None)

    st.success("Dataset uploaded successfully")

    # Preview
    st.markdown("### 👀 Preview")
    st.dataframe(data["preview"], use_container_width=True)

    st.markdown("---")
    st.info(
        f"""
        **Dataset ID:** `{data['dataset_id']}`  
        **Columns:** {len(data['columns'])}

        You can now proceed to:
        - 📊 EDA
        - 🧹 ETL
        - 🤖 Train
        """
    )
