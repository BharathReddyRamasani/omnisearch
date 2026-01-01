import streamlit as st

st.set_page_config(
    page_title="OmniSearch AI",
    layout="wide"
)

st.title("🚀 OmniSearch AI")
st.caption("EDA • ETL • AutoML • Prediction")

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

st.info("Use the sidebar to navigate through the workflow.")
