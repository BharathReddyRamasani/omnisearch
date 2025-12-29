# frontend/ETL.py - ETL DASHBOARD WITH DOWNLOADS + COMPARISON
import streamlit as st
import requests
import pandas as pd

# Your API config
API = "http://localhost:8000"
dataset_id = st.session_state.get("dataset_id", None)

st.set_page_config(layout="wide", page_title="ETL Pipeline")
st.title("🧹 ETL Pipeline - Raw vs Clean")

# ------------------ CHECK DATASET ------------------
if not dataset_id:
    st.warning("👈 **Upload dataset first** (Home page)")
    st.stop()

st.success(f"**Active Dataset:** `{dataset_id}`")

# ------------------ MAIN CONTROLS ------------------
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("🚀 **Run Full ETL Pipeline**", type="primary", use_container_width=True):
        with st.spinner("🧹 Cleaning outliers + filling missing values..."):
            resp = requests.post(f"{API}/datasets/{dataset_id}/clean")
            if resp.status_code == 200:
                st.success("✅ **ETL COMPLETE!** Clean file ready.")
                st.balloons()
                st.rerun()
            else:
                st.error(f"ETL failed: {resp.json().get('detail', 'Unknown error')}")

with col2:
    if st.button("📊 **Show Comparison**", use_container_width=True):
        st.rerun()

# ------------------ DOWNLOAD BUTTONS ------------------
st.markdown("---")
st.markdown("## ⬇️ **Download Files**")

col1, col2 = st.columns(2)

with col1:
    try:
        resp = requests.get(f"{API}/datasets/{dataset_id}/download/clean")
        if resp.status_code == 200:
            st.download_button(
                label="📥 **Download Clean CSV**",
                data=resp.content,
                file_name=f"{dataset_id}_clean.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("🧹 **Run ETL first** to download clean.csv")
    except:
        st.info("🧹 **Run ETL first**")

with col2:
    try:
        resp = requests.get(f"{API}/datasets/{dataset_id}/download/raw")
        if resp.status_code == 200:
            st.download_button(
                label="📥 **Download Raw CSV**",
                data=resp.content,
                file_name=f"{dataset_id}_raw.csv",
                mime="text/csv",
                use_container_width=True
            )
    except:
        st.info("📁 Raw file not found")

# ------------------ COMPARISON METRICS ------------------
st.markdown("---")
st.markdown("## 📊 **Raw vs Clean Comparison**")

try:
    resp = requests.get(f"{API}/datasets/{dataset_id}/comparison")
    if resp.status_code == 200:
        comp = resp.json()
        
        # BIG METRICS
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📈 Rows", f"{comp['raw_stats']['rows']:,}", f"{comp['clean_stats']['rows']:,}")
        with col2:
            st.metric("🔍 Missing %", f"{comp['raw_stats']['missing_percent']:.1f}%", f"{comp['clean_stats']['missing_percent']:.1f}%")
        with col3:
            st.metric("⚠️ Outliers Fixed", comp['improvements']['outliers_fixed'])
        with col4:
            st.metric("🎯 Expected Lift", f"+{comp['accuracy_lift_expected']:.1f}%")
        
        # SUMMARY TABLE
        st.markdown("### **Detailed Improvements**")
        st.json(comp)
        
        # DEMO MAGIC
        st.success(f"""
        ✅ **{comp['improvements']['outliers_fixed']:,} outliers clipped!**
        ✅ **{comp['improvements']['missing_values_filled']:,} missing values filled!** 
        ✅ **{comp['improvements']['numeric_columns_cleaned']} columns cleaned!**
        """)
    else:
        st.info("👈 **Run ETL first** to see comparison")
except:
    st.info("👈 **Run ETL first** to see comparison")

## ------------------ NEXT STEPS WITH CLEAN DATA ------------------
st.markdown("---")
st.markdown("## 🚀 **Next: Use Clean Data Everywhere**")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📈 **EDA (Clean Data)**", use_container_width=True):
        st.session_state.next_page = "EDA"
        st.rerun()
with col2:
    if st.button("🤖 **Train Model (Clean Data)**", use_container_width=True):
        st.session_state.next_page = "Train"
        st.rerun()
with col3:
    if st.button("🔮 **Predict**", use_container_width=True):
        st.session_state.next_page = "Predict"
        st.rerun()

# Auto-navigate
if st.session_state.get("next_page"):
    st.switch_page(f"pages/{st.session_state.next_page}.py")
