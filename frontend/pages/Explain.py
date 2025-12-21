import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8000"

st.title("🧠 Model Explainability")

if "dataset_id" not in st.session_state:
    st.stop()

meta = requests.get(
    f"{API}/meta",
    params={"dataset_id": st.session_state["dataset_id"]}
).json()

top_features = meta.get("top_features", [])

if not top_features:
    st.info("No explainability available.")
    st.stop()

st.subheader("Top Influencing Features")
df = pd.DataFrame({
    "Feature": top_features,
    "Importance Rank": list(range(1, len(top_features)+1))
})

st.dataframe(df)
