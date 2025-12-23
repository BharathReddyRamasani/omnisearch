import streamlit as st

st.set_page_config(layout="wide")
st.title("🚀 **OmniSearch AI**")

# FAST UPLOAD (1 click)
uploaded_file = st.file_uploader("📁 Drop CSV", type="csv")
if uploaded_file:
    st.session_state.dataset_id = uploaded_file.name.replace(".csv", "")
    st.success(f"✅ **{uploaded_file.name} loaded instantly!**")
    st.balloons()
