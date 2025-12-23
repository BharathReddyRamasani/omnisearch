import streamlit as st
import requests

st.title("📈 **Executive Dashboard**")

if "dataset_id" not in st.session_state:
    st.info("👆 Upload dataset to see dashboard")
    st.stop()

dataset_id = st.session_state.dataset_id

# Fetch metrics
try:
    eda_resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}")
    meta_resp = requests.get(f"http://127.0.0.1:8000/meta/{dataset_id}")
    
    if eda_resp.status_code == 200:
        eda = eda_resp.json()["eda"]
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Rows", f"{eda.get('rows', 0):,}")
        with col2: st.metric("Columns", eda.get('columns', 0))
        with col3: st.metric("Missing", sum(eda.get('missing', {}).values()))
    
    if meta_resp.status_code == 200:
        st.success("✅ **Model Trained!** Ready for production")
    
except:
    st.info("📊 Train model to unlock dashboard")

st.markdown("---")
st.success("🚀 **Production ML Pipeline Active**")
