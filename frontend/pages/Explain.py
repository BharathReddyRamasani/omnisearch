import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8003"
st.title("💡 Model Explainability")

if 'dataset_id' not in st.session_state:
    st.stop()

dataset_id = st.session_state.dataset_id

meta_resp = requests.get(f"{API}/meta/{dataset_id}")
if meta_resp.status_code != 200:
    st.warning("Train a model first")
    st.stop()

meta = meta_resp.json()
top_features = meta.get("top_features", [])
target = meta["target"]
task = meta["task"]

st.markdown(f"**🎯 Target**: {target}")
st.markdown(f"**📊 Task Type**: {task.title()}")

if not top_features:
    st.info("No explainability available for this model.")
    st.stop()

st.subheader("🏆 Top Influencing Features")
df = pd.DataFrame({
    "Feature": top_features,
    "Importance Rank": range(1, len(top_features) + 1)
})
st.dataframe(df, use_container_width=True)

st.info("💡 These features were selected based on model-derived importance, not manual selection.")
