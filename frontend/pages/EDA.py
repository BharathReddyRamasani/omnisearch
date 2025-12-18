import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("📊 Exploratory Data Analysis (EDA)")

# ------------------ CHECK DATASET ------------------
if "dataset_id" not in st.session_state:
    st.warning("⚠️ Upload dataset first")
    st.stop()

# ------------------ RUN EDA ------------------
if st.button("Run EDA"):

    resp = requests.get(
        f"{API}/eda",
        params={"dataset_id": st.session_state["dataset_id"]}
    )

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    data = resp.json()

    if data.get("status") != "ok":
        st.error(data)
        st.stop()

    eda = data["eda"]

    # ================= DATA HEALTH =================
    st.subheader("📉 Missing Values")
    st.json(eda["missing"])

    st.subheader("📊 Data Types")
    st.json(eda["dtypes"])

    # ================= BEFORE CLEANING =================
    st.markdown("---")
    st.header("🔴 Before Cleaning")

    st.subheader("📈 Summary Statistics (Raw Data)")
    st.json(eda["before"]["summary"])

    st.subheader("🚨 Outlier Detection (IQR)")
    st.json(eda["before"]["outliers"])

    if eda["before"]["plots"]:
        st.subheader("📊 Distributions & Outliers (Before)")
        for col, img in eda["before"]["plots"].items():
            st.image(img, caption=f"{col} — Before Cleaning", use_container_width=True)
    else:
        st.info("No numeric columns found for plotting (Before)")

    # ================= AFTER CLEANING =================
    st.markdown("---")
    st.header("🟢 After Cleaning")

    st.subheader("📈 Summary Statistics (Cleaned Data)")
    st.json(eda["after"]["summary"])

    if eda["after"]["plots"]:
        st.subheader("📊 Distributions (After Outlier Handling)")
        for col, img in eda["after"]["plots"].items():
            st.image(img, caption=f"{col} — After Cleaning", use_container_width=True)
    else:
        st.info("No numeric columns found for plotting (After)")
