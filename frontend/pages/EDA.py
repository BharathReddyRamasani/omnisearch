import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="OmniSearch AI - Enterprise EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DATASET CHECK ----------
if "dataset_id" not in st.session_state or not st.session_state.dataset_id:
    st.error("🚫 No dataset loaded. Go to **Upload** page first.")
    st.stop()

dataset_id = st.session_state.dataset_id

# ---------- HEADER ----------
st.markdown(
    """
<div style='background: linear-gradient(90deg, #1e3a72 0%, #2c5aa0 100%);
            padding: 2.5rem; border-radius: 18px; color: white;
            text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <h1 style='margin: 0; font-size: 3.2rem;'>📊 <b>Enterprise EDA Engine</b></h1>
    <p style='margin-top: 8px; font-size: 1.3rem; opacity: 0.9;'>
        ETL Intelligence • Production Analytics • Data Quality Scoring
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🎛️ EDA Controls")

    if st.button("🔄 Refresh Analysis", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("eda_"):
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Quick Status")
    if st.session_state.get("eda_data"):
        quick = st.session_state.eda_data
        st.metric("Rows", f"{quick.get('rows', 0):,}")
        st.metric("Quality", f"{quick.get('quality_score', 0):.0f}%")
        st.metric("Source", quick.get("data_source", "RAW"))

# ---------- RUN BUTTON ----------
run_clicked = st.button(
    "🚀 Run Enterprise EDA Pipeline",
    type="primary",
    use_container_width=True,
)

# ---------- USAGE TEXT (NO AUTO RUN) ----------
if "eda_data" not in st.session_state and not run_clicked:
    st.markdown(
        """
### 🚀 Enterprise EDA Ready

**Behavior:**
- If **ETL not run** → analyzes **RAW data**
- If **ETL completed** → analyzes **CLEAN data**

**Computes:**
- Missing value profile
- Data quality score (0–100)
- Statistical summary
- ETL impact (if available)

Click **Run Enterprise EDA Pipeline** to begin.
"""
    )
    st.stop()

# ---------- CALL BACKEND ----------
if run_clicked:
    with st.spinner("🔬 Running EDA (backend auto-selects RAW vs CLEAN)..."):
        try:
            resp = requests.get(f"{API}/eda/{dataset_id}", timeout=60)
            if resp.status_code != 200:
                st.error(f"Backend error: {resp.status_code}")
                st.stop()

            payload = resp.json()
            if payload.get("status") != "ok":
                st.error("EDA failed")
                st.stop()

            st.session_state.eda_data = payload["eda"]
            st.success("✅ Enterprise EDA Complete")
            st.rerun()

        except Exception as e:
            st.error(f"Connection error: {str(e)}")
            st.stop()

# ---------- DISPLAY RESULTS ----------
eda = st.session_state.get("eda_data")
if not eda:
    st.stop()

# ---------- EXECUTIVE SUMMARY ----------
st.markdown("### 🎯 Executive Summary")
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.metric("Rows", f"{eda['rows']:,}")
with c2:
    st.metric("Columns", eda["columns"])
with c3:
    st.metric("Missing %", f"{eda['missing_pct']:.1f}%")
with c4:
    st.metric("Quality", f"{eda['quality_score']:.0f}%")
with c5:
    st.metric("Source", eda.get("data_source", "RAW"))
with c6:
    st.metric("ETL", "CLEAN" if eda.get("etl_complete") else "RAW")

# ---------- ETL INTELLIGENCE ----------
st.markdown("### 🧹 ETL Intelligence")
if eda.get("etl_complete") and eda.get("etl_improvements"):
    imp = eda["etl_improvements"]["improvements"]
    lift = eda["etl_improvements"]["accuracy_lift_expected"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Outliers Fixed", f"{imp['outliers_fixed']:,}")
    with c2:
        st.metric("Missing Filled", f"{imp['missing_values_filled']:,}")
    with c3:
        st.metric("Expected Lift", f"+{lift:.1f}%")
else:
    st.info("ETL not run yet – analysis based on RAW data.")

# ---------- QUALITY GAUGE ----------
st.markdown("### ⚡ Data Quality Gauge")
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=eda["quality_score"],
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2c5aa0"},
            "steps": [
                {"range": [0, 70], "color": "red"},
                {"range": [70, 90], "color": "orange"},
                {"range": [90, 100], "color": "green"},
            ],
        },
        title={"text": "Data Quality Score"},
    )
)
st.plotly_chart(fig_gauge, use_container_width=True)

# ---------- MISSING VALUES ----------
st.markdown("### 🚫 Missing Value Intelligence")
if eda.get("missing"):
    missing_df = (
        pd.DataFrame(eda["missing"].items(), columns=["Column", "Missing"])
        .assign(Percent=lambda x: (x["Missing"] / eda["rows"] * 100).round(2))
        .sort_values("Percent", ascending=False)
        .head(15)
    )

    fig = px.bar(
        missing_df,
        x="Column",
        y="Percent",
        color="Percent",
        color_continuous_scale="Reds",
        title="Top 15 Columns by Missing %",
    )
    fig.update_layout(xaxis_tickangle=-45, height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing values detected.")

# ---------- STATISTICAL PROFILE ----------
st.markdown("### 📈 Statistical Profile")
if eda.get("summary"):
    summary_df = pd.DataFrame(eda["summary"]).T.round(2)
    st.dataframe(summary_df, use_container_width=True)

# ---------- EXPORTS ----------
st.markdown("### 📥 Exports")
c1, c2 = st.columns(2)

with c1:
    st.download_button(
        "📄 Download JSON Report",
        data=json.dumps(eda, indent=2),
        file_name=f"EDA_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )

with c2:
    st.download_button(
        "📊 Download Summary CSV",
        data=pd.DataFrame([eda]).to_csv(index=False),
        file_name=f"EDA_Summary_{dataset_id}.csv",
        mime="text/csv",
    )
