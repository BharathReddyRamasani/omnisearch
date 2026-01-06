# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import json
# from datetime import datetime

# API = "http://127.0.0.1:8000/api"

# st.set_page_config(
#     page_title="OmniSearch AI - Enterprise EDA",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------- DATASET CHECK ----------
# if "dataset_id" not in st.session_state or not st.session_state.dataset_id:
#     st.error("🚫 No dataset loaded. Go to **Upload** page first.")
#     st.stop()

# dataset_id = st.session_state.dataset_id

# # ---------- HEADER ----------
# st.markdown(
#     """
# <div style='background: linear-gradient(90deg, #1e3a72 0%, #2c5aa0 100%);
#             padding: 2.5rem; border-radius: 18px; color: white;
#             text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
#     <h1 style='margin: 0; font-size: 3.2rem;'>📊 <b>Enterprise EDA Engine</b></h1>
#     <p style='margin-top: 8px; font-size: 1.3rem; opacity: 0.9;'>
#         ETL Intelligence • Production Analytics • Data Quality Scoring
#     </p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# # ---------- SIDEBAR ----------
# with st.sidebar:
#     st.markdown("### 🎛️ EDA Controls")

#     if st.button("🔄 Refresh Analysis", use_container_width=True):
#         for key in list(st.session_state.keys()):
#             if key.startswith("eda_"):
#                 del st.session_state[key]
#         st.rerun()

#     st.markdown("---")
#     st.markdown("### 📊 Quick Status")
#     if st.session_state.get("eda_data"):
#         quick = st.session_state.eda_data
#         st.metric("Rows", f"{quick.get('rows', 0):,}")
#         st.metric("Quality", f"{quick.get('quality_score', 0):.0f}%")
#         st.metric("Source", quick.get("data_source", "RAW"))

# # ---------- RUN BUTTON ----------
# run_clicked = st.button(
#     "🚀 Run Enterprise EDA Pipeline",
#     type="primary",
#     use_container_width=True,
# )

# # ---------- USAGE TEXT (NO AUTO RUN) ----------
# if "eda_data" not in st.session_state and not run_clicked:
#     st.markdown(
#         """
# ### 🚀 Enterprise EDA Ready

# **Behavior:**
# - If **ETL not run** → analyzes **RAW data**
# - If **ETL completed** → analyzes **CLEAN data**

# **Computes:**
# - Missing value profile
# - Data quality score (0–100)
# - Statistical summary
# - ETL impact (if available)

# Click **Run Enterprise EDA Pipeline** to begin.
# """
#     )
#     st.stop()

# # ---------- CALL BACKEND ----------
# if run_clicked:
#     with st.spinner("🔬 Running EDA (backend auto-selects RAW vs CLEAN)..."):
#         try:
#             resp = requests.get(f"{API}/eda/{dataset_id}", timeout=60)
#             if resp.status_code != 200:
#                 st.error(f"Backend error: {resp.status_code}")
#                 st.stop()

#             payload = resp.json()
#             if payload.get("status") != "ok":
#                 st.error("EDA failed")
#                 st.stop()

#             st.session_state.eda_data = payload["eda"]
#             st.success("✅ Enterprise EDA Complete")
#             st.rerun()

#         except Exception as e:
#             st.error(f"Connection error: {str(e)}")
#             st.stop()

# # ---------- DISPLAY RESULTS ----------
# eda = st.session_state.get("eda_data")
# if not eda:
#     st.stop()

# # ---------- EXECUTIVE SUMMARY ----------
# st.markdown("### 🎯 Executive Summary")
# c1, c2, c3, c4, c5, c6 = st.columns(6)

# with c1:
#     st.metric("Rows", f"{eda['rows']:,}")
# with c2:
#     st.metric("Columns", eda["columns"])
# with c3:
#     st.metric("Missing %", f"{eda['missing_pct']:.1f}%")
# with c4:
#     st.metric("Quality", f"{eda['quality_score']:.0f}%")
# with c5:
#     st.metric("Source", eda.get("data_source", "RAW"))
# with c6:
#     st.metric("ETL", "CLEAN" if eda.get("etl_complete") else "RAW")

# # ---------- ETL INTELLIGENCE ----------
# st.markdown("### 🧹 ETL Intelligence")
# if eda.get("etl_complete") and eda.get("etl_improvements"):
#     imp = eda["etl_improvements"]["improvements"]
#     lift = eda["etl_improvements"]["accuracy_lift_expected"]

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.metric("Outliers Fixed", f"{imp['outliers_fixed']:,}")
#     with c2:
#         st.metric("Missing Filled", f"{imp['missing_values_filled']:,}")
#     with c3:
#         st.metric("Expected Lift", f"+{lift:.1f}%")
# else:
#     st.info("ETL not run yet – analysis based on RAW data.")

# # ---------- QUALITY GAUGE ----------
# st.markdown("### ⚡ Data Quality Gauge")
# fig_gauge = go.Figure(
#     go.Indicator(
#         mode="gauge+number",
#         value=eda["quality_score"],
#         gauge={
#             "axis": {"range": [0, 100]},
#             "bar": {"color": "#2c5aa0"},
#             "steps": [
#                 {"range": [0, 70], "color": "red"},
#                 {"range": [70, 90], "color": "orange"},
#                 {"range": [90, 100], "color": "green"},
#             ],
#         },
#         title={"text": "Data Quality Score"},
#     )
# )
# st.plotly_chart(fig_gauge, use_container_width=True)

# # ---------- MISSING VALUES ----------
# st.markdown("### 🚫 Missing Value Intelligence")
# if eda.get("missing"):
#     missing_df = (
#         pd.DataFrame(eda["missing"].items(), columns=["Column", "Missing"])
#         .assign(Percent=lambda x: (x["Missing"] / eda["rows"] * 100).round(2))
#         .sort_values("Percent", ascending=False)
#         .head(15)
#     )

#     fig = px.bar(
#         missing_df,
#         x="Column",
#         y="Percent",
#         color="Percent",
#         color_continuous_scale="Reds",
#         title="Top 15 Columns by Missing %",
#     )
#     fig.update_layout(xaxis_tickangle=-45, height=400)
#     st.plotly_chart(fig, use_container_width=True)
# else:
#     st.success("No missing values detected.")

# # ---------- STATISTICAL PROFILE ----------
# st.markdown("### 📈 Statistical Profile")
# if eda.get("summary"):
#     summary_df = pd.DataFrame(eda["summary"]).T.round(2)
#     st.dataframe(summary_df, use_container_width=True)

# # ---------- EXPORTS ----------
# st.markdown("### 📥 Exports")
# c1, c2 = st.columns(2)

# with c1:
#     st.download_button(
#         "📄 Download JSON Report",
#         data=json.dumps(eda, indent=2),
#         file_name=f"EDA_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
#         mime="application/json",
#     )

# with c2:
#     st.download_button(
#         "📊 Download Summary CSV",
#         data=pd.DataFrame([eda]).to_csv(index=False),
#         file_name=f"EDA_Summary_{dataset_id}.csv",
#         mime="text/csv",
#     )







import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from io import StringIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="OmniSearch AI – Enterprise EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# DATASET CHECK
# =====================================================
if "dataset_id" not in st.session_state:
    st.error("🚫 No dataset loaded. Upload a dataset first.")
    st.stop()

dataset_id = st.session_state.dataset_id

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
<div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
padding:2.5rem;border-radius:18px;color:white;text-align:center;">
<h1 style="margin:0;font-size:3rem;">📊 Enterprise EDA Intelligence</h1>
<p style="margin-top:8px;font-size:1.2rem;opacity:.9;">
Outliers • Clustering • Feature Impact • Executive Analytics
</p>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🎛 Controls")
    run_eda = st.button("🚀 Run / Refresh EDA", use_container_width=True)
    st.markdown("---")
    export_html = st.button("🌐 Generate Executive HTML")
    export_pdf = st.button("📄 Generate Executive PDF")

# =====================================================
# BACKEND FETCHERS
# =====================================================
@st.cache_data(ttl=60)
def fetch_eda(ds):
    r = requests.get(f"{API}/eda/{ds}", timeout=60)
    r.raise_for_status()
    return r.json()["eda"]

@st.cache_data(ttl=60)
def fetch_etl(ds):
    r = requests.get(f"{API}/datasets/{ds}/comparison", timeout=30)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=60)
def fetch_model_meta(ds):
    r = requests.get(f"{API}/meta/{ds}", timeout=10)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=60)
def load_raw_sample(ds, cols=None, n=50000):
    r = requests.get(f"{API}/datasets/{ds}/download/raw", timeout=60)
    df = pd.read_csv(StringIO(r.text), usecols=cols)
    return df.sample(min(len(df), n), random_state=42)

# =====================================================
# LOAD DATA
# =====================================================
if run_eda or "eda" not in st.session_state:
    with st.spinner("🔬 Running Enterprise EDA…"):
        st.session_state.eda = fetch_eda(dataset_id)
        st.session_state.etl = fetch_etl(dataset_id)
        st.session_state.model_meta = fetch_model_meta(dataset_id)

eda = st.session_state.eda
etl = st.session_state.etl
model_meta = st.session_state.model_meta

# =====================================================
# EXECUTIVE KPI DASHBOARD
# =====================================================
st.markdown("## 🎯 Executive KPI Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{eda['rows']:,}")
c2.metric("Columns", eda["columns"])
c3.metric("Missing %", f"{eda['missing_pct']:.1f}%")
c4.metric("Quality Score", f"{eda['quality_score']:.0f}/100")
c5.metric("Source", "CLEAN" if eda.get("etl_complete") else "RAW")

# =====================================================
# BEFORE / AFTER COMPARISON
# =====================================================
st.markdown("## 🔄 RAW vs CLEAN Comparison")

if etl:
    raw = etl["comparison"]["raw_stats"]
    clean = etl["comparison"]["clean_stats"]
    quality = etl["comparison"]["quality"]

    cmp = pd.DataFrame(
        [
            ["Rows", raw["rows"], clean["rows"]],
            ["Missing Values", raw["missing_total"], clean["missing_total"]],
            ["Duplicates", raw["duplicate_rows"], clean["duplicate_rows"]],
            ["Quality Score", quality["before"], quality["after"]],
        ],
        columns=["Metric", "RAW", "CLEAN"],
    )
    st.dataframe(cmp, use_container_width=True)
else:
    st.info("ETL not run yet.")

# =====================================================
# QUALITY GAUGE
# =====================================================
st.markdown("## ⚡ Data Quality Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=eda["quality_score"],
    gauge={
        "axis": {"range": [0, 100]},
        "steps": [
            {"range": [0, 70], "color": "red"},
            {"range": [70, 90], "color": "orange"},
            {"range": [90, 100], "color": "green"},
        ],
    },
))
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MISSING VALUE HEATMAP
# =====================================================
st.markdown("## 🚫 Missing Value Intelligence")

missing_df = (
    pd.DataFrame(eda["missing"].items(), columns=["Feature", "Missing"])
    .assign(Percent=lambda x: x["Missing"] / eda["rows"] * 100)
    .sort_values("Percent", ascending=False)
)

fig = px.bar(
    missing_df.head(15),
    x="Feature",
    y="Percent",
    color="Percent",
    color_continuous_scale="Reds",
)
st.plotly_chart(fig, use_container_width=True)

# =====================================================
# BOXPLOTS + OUTLIERS
# =====================================================
st.markdown("## 📦 Outlier Analysis (Boxplots)")

num_features = [
    f for f, v in eda["summary"].items()
    if isinstance(v.get("mean"), (int, float))
]

if num_features:
    feature = st.selectbox("Select numeric feature", num_features)
    df_box = load_raw_sample(dataset_id, [feature])

    fig = px.box(
        df_box,
        y=feature,
        points="outliers",
        title=f"Boxplot – {feature}",
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# CLUSTERING (UNSUPERVISED)
# =====================================================
st.markdown("## 🧠 Unsupervised Clustering (Outliers + Structure)")

if len(num_features) >= 2:
    x_col, y_col = st.multiselect(
        "Select 2 features for clustering",
        num_features,
        default=num_features[:2],
        max_selections=2,
    )

    if len([x_col, y_col]) == 2:
        df_cluster = load_raw_sample(dataset_id, [x_col, y_col])
        X = StandardScaler().fit_transform(df_cluster)

        algo = st.radio("Clustering Algorithm", ["KMeans", "DBSCAN"])

        if algo == "KMeans":
            k = st.slider("Clusters (k)", 2, 8, 3)
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
        else:
            labels = DBSCAN(eps=0.8, min_samples=10).fit_predict(X)

        df_cluster["cluster"] = labels

        fig = px.scatter(
            df_cluster,
            x=x_col,
            y=y_col,
            color="cluster",
            title=f"{algo} Clustering ({x_col} vs {y_col})",
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# EXECUTIVE EXPORTS
# =====================================================
st.markdown("## 📥 Executive Reports")

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
json_report = json.dumps(
    {
        "eda": eda,
        "etl": etl,
        "model": model_meta,
    },
    indent=2,
)

if export_html:
    html = f"""
    <h1>Enterprise EDA Report</h1>
    <pre>{json_report}</pre>
    """
    st.download_button(
        "Download HTML Report",
        html,
        f"EDA_Report_{dataset_id}_{timestamp}.html",
        "text/html",
    )

if export_pdf:
    st.info("PDF generated from HTML (wkhtmltopdf recommended in prod).")
    st.download_button(
        "Download PDF Source",
        json_report,
        f"EDA_Report_{dataset_id}_{timestamp}.pdf",
        "application/pdf",
    )

st.download_button(
    "Download JSON",
    json_report,
    f"EDA_Report_{dataset_id}.json",
    "application/json",
)

st.caption("Enterprise EDA • Unsupervised Intelligence • Audit-Ready")
