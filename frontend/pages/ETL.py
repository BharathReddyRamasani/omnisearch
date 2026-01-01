import streamlit as st
import requests

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    layout="wide",
    page_title="OmniSearch AI — Enterprise ETL"
)

st.markdown("## 🧹 Enterprise ETL Pipeline")

# ---------------- DATASET VALIDATION ----------------
dataset_id = st.session_state.get("dataset_id")

if not dataset_id:
    st.error("No dataset loaded. Upload a dataset first.")
    st.stop()

st.success(f"Active Dataset: `{dataset_id}`")

# ---------------- ETL EXECUTION ----------------
col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 Execute Enterprise ETL", type="primary", use_container_width=True):
        with st.spinner("Running ETL (dedup • imputation • outliers • quality scoring)…"):
            r = requests.post(f"{API}/datasets/{dataset_id}/clean")

            if r.status_code == 200:
                st.session_state.etl_done = True
                st.success("ETL completed successfully")
                st.rerun()
            else:
                if r.headers.get("content-type", "").startswith("application/json"):
                    st.error(r.json().get("detail", "ETL failed"))
                else:
                    st.error("ETL failed — non-JSON backend response")
                    st.code(r.text)

with col2:
    if st.button("📊 Load Comparison", use_container_width=True):
        st.session_state.show_comp = True
        st.rerun()

# ---------------- DOWNLOAD ZONE ----------------
st.markdown("---")
st.markdown("## ⬇️ Dataset Artifacts")

d1, d2 = st.columns(2)

# CLEAN
with d1:
    r = requests.get(f"{API}/datasets/{dataset_id}/download/clean")
    if r.status_code == 200:
        st.download_button(
            "Download Clean Dataset",
            data=r.content,
            file_name=f"{dataset_id}_clean.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Clean dataset not available (ETL not run yet)")

# RAW
with d2:
    r = requests.get(f"{API}/datasets/{dataset_id}/download/raw")
    if r.status_code == 200:
        st.download_button(
            "Download Raw Dataset",
            data=r.content,
            file_name=f"{dataset_id}_raw.csv",
            mime="text/csv",
            use_container_width=True
        )

# ---------------- COMPARISON ----------------
st.markdown("---")
st.markdown("## 📊 ETL Impact Report")

if st.session_state.get("show_comp") or st.session_state.get("etl_done"):
    r = requests.get(f"{API}/datasets/{dataset_id}/comparison")

    if r.status_code != 200:
        st.warning("Run ETL to generate comparison report")
        st.stop()

    comp = r.json()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", comp["raw_stats"]["rows"], comp["clean_stats"]["rows"])
    c2.metric("Missing Filled", comp["improvements"]["missing_values_filled"])
    c3.metric("Outliers Fixed", comp["improvements"]["outliers_fixed"])
    c4.metric("Expected Lift", f"+{comp['accuracy_lift_expected']}%")

    st.markdown("### Detailed ETL Intelligence")
    st.json(comp)

# ---------------- NEXT STEPS ----------------
st.markdown("---")
st.markdown("## 🚀 Production Flow")

n1, n2, n3 = st.columns(3)

with n1:
    if st.button("📈 EDA (Clean)", use_container_width=True):
        st.switch_page("pages/EDA.py")


with n2:
    if st.button("Train Model", use_container_width=True):
        st.switch_page("pages/Train.py")
with n3:
    if st.button("Predict", use_container_width=True):
        st.switch_page("pages/Predict.py")
