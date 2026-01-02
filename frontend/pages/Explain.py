import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API = "http://127.0.0.1:8000/api"
dataset_id = st.session_state.get("dataset_id")

st.title("🔍 Model Explainability (SHAP)")

resp = requests.get(f"{API}/explain/{dataset_id}")
if resp.status_code != 200:
    st.error("Run training first")
    st.stop()

data = resp.json()

df = pd.DataFrame({
    "Feature": data["features"],
    "Importance": data["importance"]
}).sort_values("Importance", ascending=False).head(20)

fig = px.bar(
    df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top 20 Feature Importances (SHAP)"
)

st.plotly_chart(fig, use_container_width=True)
