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

    eda = data.get("eda", {})

    # ================= DATA HEALTH =================
    st.subheader("📉 Missing Values")
    st.json(eda.get("missing", {}))

    st.subheader("📊 Data Types")
    st.json(eda.get("dtypes", {}))

    # ---------------- SAFE FETCH ----------------
    before = eda.get("before", {})
    after = eda.get("after", {})

    before_summary = before.get("summary", {})
    before_outliers = before.get("outliers", {})
    before_plots = before.get("plots", {})

    after_summary = after.get("summary", {})
    after_plots = after.get("plots", {})

    # ================= BEFORE CLEANING =================
    st.markdown("---")
    st.header("🔴 Before Cleaning")

    if before_summary:
        st.subheader("📈 Summary Statistics (Raw Data)")
        st.json(before_summary)
    else:
        st.info("No summary statistics available (Before Cleaning)")

    if before_outliers:
        st.subheader("🚨 Outlier Detection (IQR)")
        st.json(before_outliers)
    else:
        st.info("No outlier information available")

    if before_plots:
        st.subheader("📊 Distributions (Before Cleaning)")
        for col, img in before_plots.items():
            st.image(
                img,
                caption=f"{col} — Before Cleaning",
                use_container_width=True
            )
    else:
        st.info("No numeric columns found for plotting (Before)")

    # ================= AFTER CLEANING =================
    st.markdown("---")
    st.header("🟢 After Cleaning")

    if after_summary:
        st.subheader("📈 Summary Statistics (Cleaned Data)")
        st.json(after_summary)
    else:
        st.info("No summary statistics available (After Cleaning)")

    if after_plots:
        st.subheader("📊 Distributions (After Outlier Handling)")
        for col, img in after_plots.items():
            st.image(
                img,
                caption=f"{col} — After Cleaning",
                use_container_width=True
            )
    else:
        st.info("No numeric columns found for plotting (After)")
