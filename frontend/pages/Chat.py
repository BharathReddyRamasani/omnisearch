import streamlit as st
import pandas as pd

st.title("💬 Ask Your Data")

if "dataset_id" not in st.session_state:
    st.stop()

df = pd.read_csv(
    f"data/datasets/{st.session_state['dataset_id']}/raw.csv"
)

question = st.text_input("Ask a question about your dataset")

if question:
    q = question.lower()

    if "average" in q or "mean" in q:
        st.write(df.describe())

    elif "missing" in q or "null" in q:
        st.write(df.isnull().sum())

    elif "top" in q:
        st.write(df.head())

    else:
        st.info(
            "I can answer questions about averages, missing values, "
            "top rows, and distributions."
        )
