import streamlit as st

st.set_page_config(page_title="OmniSearch AI", layout="wide")

st.title("🔍 OmniSearch AI – CSV Intelligence Workbench")

st.markdown("""
**End-to-End Data Workflow**
- Upload dirty CSV
- Automatic EDA
- Train ML models
- Predict using trained model
""")

if "dataset_id" in st.session_state:
    st.success(f"Active Dataset: {st.session_state['dataset_id']}")
else:
    st.warning("No dataset uploaded yet. Go to Upload page.")

st.markdown("""
### Workflow
1️⃣ Upload CSV  
2️⃣ Run EDA  
3️⃣ Train Model  
4️⃣ Predict  
""")
