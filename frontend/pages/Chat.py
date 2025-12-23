import streamlit as st
import requests

API = "http://127.0.0.1:8000"
st.title("💬 Chat With Your Data")

if 'dataset_id' not in st.session_state:
    st.warning("Upload dataset first")
    st.stop()

dataset_id = st.session_state.dataset_id

st.info("🤖 Ask about: model accuracy, top features, performance, or model selection")

question = st.text_input("Ask a question about your dataset:")

if question:
    q = question.lower()
    if "average" in q or "mean" in q:
        st.write("📊 **Averages**: Check EDA page for detailed statistics")
    elif "missing" in q or "null" in q:
        st.write("🔍 **Missing Values**: See EDA report for missing value counts")
    elif "top" in q:
        st.write("⭐ **Top Features**: Check Explainability page")
    else:
        resp = requests.post(f"{API}/chat/{dataset_id}", json={"question": question})
        result = resp.json()
        st.write(f"**🤖 Answer**: {result.get('answer', 'No answer available')}")
        
        if "data" in result:
            st.json(result["data"])
