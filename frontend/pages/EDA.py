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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

API = "http://127.0.0.1:8003/api"

st.set_page_config(
    page_title="OmniSearch AI – Enterprise EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# SESSION STATE INITIALIZATION (CRITICAL - MUST BE FIRST)
# =====================================================
for key in ["eda", "etl", "model_meta"]:
    if key not in st.session_state:
        st.session_state[key] = None

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
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Parameters")
    sample_size = st.slider("Sample Size for Analysis", 1000, 50000, 10000, step=1000)
    random_state = st.slider("Random State", 0, 100, 42)

# =====================================================
# BACKEND FETCHERS
# =====================================================
@st.cache_data(ttl=300)
def fetch_eda(ds):
    r = requests.get(f"{API}/eda/{ds}", timeout=180)
    r.raise_for_status()
    return r.json()["eda"]

@st.cache_data(ttl=300)
def fetch_etl(ds):
    r = requests.get(f"{API}/datasets/{ds}/comparison", timeout=60)
    return r.json() if r.status_code == 200 else None

@st.cache_data(ttl=300)
def fetch_model_meta(ds):
    r = requests.get(f"{API}/meta/{ds}", timeout=20)
    return r.json() if r.status_code == 200 else None

# =====================================================
# SAFE SAMPLE LOADER — CLEAN-AWARE, CASE-INSENSITIVE
# =====================================================
@st.cache_data(ttl=300)
def load_sample(ds, cols=None, n=20000):
    # Use cached eda to determine source
    eda_data = fetch_eda(ds)
    source = "clean" if eda_data.get("etl_complete", False) else "raw"

    r = requests.get(f"{API}/datasets/{ds}/download/{source}", timeout=120)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text))
    df.columns = [c.lower() for c in df.columns]  # Normalize once

    if cols:
        lowered_cols = [c.lower() for c in cols]
        available = [c for c in lowered_cols if c in df.columns]
        if not available:
            return pd.DataFrame()  # Empty if nothing matches
        df = df[available]

    if len(df) > n:
        df = df.sample(n, random_state=42)

    return df

# =====================================================
# LOAD / REFRESH EDA DATA
# =====================================================
if run_eda:
    with st.spinner("🔬 Running Enterprise EDA…"):
        try:
            st.session_state.eda = fetch_eda(dataset_id)
            st.session_state.etl = fetch_etl(dataset_id)
            st.session_state.model_meta = fetch_model_meta(dataset_id)
        except Exception as e:
            st.error(f"Failed to fetch EDA data: {str(e)}")
            if st.session_state.eda is None:
                st.stop()

eda = st.session_state.eda
etl = st.session_state.etl
model_meta = st.session_state.model_meta

if eda is None:
    st.info("Click '🚀 Run / Refresh EDA' to start the EDA process.")
    st.stop()

# =====================================================
# MAIN ANALYSIS TABS
# =====================================================
tabs = st.tabs([
    "📊 Overview",
    "🧠 Clustering",
    "📉 Dimensionality Reduction",
    "🔍 Anomaly Detection",
    "📦 Outlier Analysis",
    "🔗 Feature Analysis",
    "📥 Exports"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================
with tabs[0]:
    st.markdown("## 🎯 Executive KPI Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{eda['rows']:,}")
    c2.metric("Columns", eda["columns"])
    c3.metric("Missing %", f"{eda['missing_pct']:.1f}%")
    c4.metric("Quality Score", f"{eda['quality_score']:.0f}/100")
    c5.metric("Source", "CLEAN" if eda.get("etl_complete") else "RAW")

    # RAW vs CLEAN Comparison
    st.markdown("## 🔄 RAW vs CLEAN Comparison")

    if etl and "comparison" in etl:
        comp = etl["comparison"]
        if comp and "raw_stats" in comp and "clean_stats" in comp:
            raw = comp["raw_stats"]
            clean = comp["clean_stats"]
            quality = comp.get("quality", {})

            cmp = pd.DataFrame(
                [
                    ["Rows", raw.get("rows", 0), clean.get("rows", 0)],
                    ["Missing Values", raw.get("missing_total", 0), clean.get("missing_total", 0)],
                    ["Duplicates", raw.get("duplicate_rows", 0), clean.get("duplicate_rows", 0)],
                    ["Quality Score", quality.get("before", 0), quality.get("after", 0)],
                ],
                columns=["Metric", "RAW", "CLEAN"],
            )
            st.dataframe(cmp, use_container_width=True)
        else:
            st.info("ℹ️ ETL not run yet — run cleaning to see improvements.")
    else:
        st.info("ℹ️ ETL not run yet — run cleaning to see improvements.")

    # Model Insights
    if model_meta and model_meta.get("status") == "ok":
        st.markdown("## 🎯 Model-Driven Insights")
        st.write(f"**Best Model:** {model_meta['best_model']} ({model_meta['best_score']:.3f})")
        if "top_features" in model_meta:
            st.write("**Top Impact Features:**", ", ".join(model_meta["top_features"]))
    else:
        st.info("ℹ️ Train a model to unlock feature importance and prediction insights.")

    # Quality Gauge
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

    # Missing Value Intelligence
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
# TAB 2: CLUSTERING
# =====================================================
with tabs[1]:
    st.markdown("## 🧠 Unsupervised Clustering")

    num_features = [
        f for f, v in eda["summary"].items()
        if v is not None and isinstance(v.get("mean"), (int, float))
    ]

    if len(num_features) >= 2:
        selected = st.multiselect(
            "Select features for clustering (2-5 recommended)",
            num_features,
            default=num_features[: min(3, len(num_features))],
            max_selections=5,
        )

        if len(selected) >= 2:
            df_cluster = load_sample(dataset_id, [f.lower() for f in selected], sample_size)

            if df_cluster.empty:
                st.warning("Selected features not available in data.")
            else:
                # ⚠️ HANDLE NaN BEFORE SCALING - CRITICAL FOR SKLEARN
                df_cluster = df_cluster.dropna()
                if df_cluster.empty:
                    st.error("All rows contain missing values. Cannot proceed.")
                    st.stop()

                # Preprocessing
                scaler_option = st.selectbox("Scaler", ["Standard", "MinMax", "Robust"], index=0)
                scaler = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}[scaler_option]
                X_scaled = scaler.fit_transform(df_cluster)

                algo = st.selectbox("Clustering Algorithm", ["KMeans", "DBSCAN", "Agglomerative", "Spectral"])

                if algo == "KMeans":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    model = KMeans(n_clusters=n_clusters, random_state=random_state)
                elif algo == "DBSCAN":
                    eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
                    min_samples = st.slider("Min Samples", 5, 20, 5)
                    model = DBSCAN(eps=eps, min_samples=min_samples)
                elif algo == "Agglomerative":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    linkage = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
                    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                elif algo == "Spectral":
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state)

                if st.button("Run Clustering"):
                    with st.spinner("Clustering..."):
                        labels = model.fit_predict(X_scaled)
                        df_cluster["cluster"] = labels

                        # Visualization (2D projection if >2 features)
                        if len(selected) == 2:
                            fig = px.scatter(
                                df_cluster,
                                x=selected[0].lower(),
                                y=selected[1].lower(),
                                color="cluster",
                                title=f"{algo} Clustering",
                            )
                        else:
                            # PCA for 2D projection
                            pca = PCA(n_components=2, random_state=random_state)
                            X_pca = pca.fit_transform(X_scaled)
                            df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                            df_pca["cluster"] = labels
                            fig = px.scatter(
                                df_pca,
                                x="PC1",
                                y="PC2",
                                color="cluster",
                                title=f"{algo} Clustering (PCA Projection)",
                            )
                        st.plotly_chart(fig, use_container_width=True)

                        # Cluster summary
                        cluster_summary = df_cluster.groupby("cluster").agg({
                            col.lower(): ["mean", "std", "count"] for col in selected
                        }).round(2)
                        st.dataframe(cluster_summary)
    else:
        st.info("Need at least 2 numeric features for clustering.")

# =====================================================
# TAB 3: DIMENSIONALITY REDUCTION
# =====================================================
with tabs[2]:
    st.markdown("## 📉 Dimensionality Reduction")

    num_features = [
        f for f, v in eda["summary"].items()
        if v is not None and isinstance(v.get("mean"), (int, float))
    ]

    if len(num_features) >= 2:
        selected = st.multiselect(
            "Select features for reduction",
            num_features,
            default=num_features[: min(5, len(num_features))],
            max_selections=10,
        )

        if len(selected) >= 2:
            df_dr = load_sample(dataset_id, [f.lower() for f in selected], sample_size)

            if df_dr.empty:
                st.warning("Selected features not available.")
            else:
                # ⚠️ HANDLE NaN BEFORE SCALING - CRITICAL FOR SKLEARN
                df_dr = df_dr.dropna()
                if df_dr.empty:
                    st.error("All rows contain missing values. Cannot proceed.")
                    st.stop()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_dr)

                method = st.selectbox("Reduction Method", ["PCA", "t-SNE"])

                if method == "PCA":
                    n_components = st.slider("Components", 2, min(5, len(selected)), 2)
                    model = PCA(n_components=n_components, random_state=random_state)
                    X_reduced = model.fit_transform(X_scaled)
                    explained_var = model.explained_variance_ratio_
                    st.write(f"Explained Variance: {explained_var}")
                    fig = px.scatter(
                        x=X_reduced[:, 0],
                        y=X_reduced[:, 1],
                        title="PCA Projection",
                        labels={"x": "PC1", "y": "PC2"}
                    )
                elif method == "t-SNE":
                    perplexity = st.slider("Perplexity", 5, 50, 30)
                    model = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
                    X_reduced = model.fit_transform(X_scaled)
                    fig = px.scatter(
                        x=X_reduced[:, 0],
                        y=X_reduced[:, 1],
                        title="t-SNE Projection",
                        labels={"x": "Dim1", "y": "Dim2"}
                    )

                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric features.")

# =====================================================
# TAB 4: ANOMALY DETECTION
# =====================================================
with tabs[3]:
    st.markdown("## 🔍 Anomaly Detection")

    num_features = [
        f for f, v in eda["summary"].items()
        if v is not None and isinstance(v.get("mean"), (int, float))
    ]

    if num_features:
        selected = st.multiselect(
            "Select features for anomaly detection",
            num_features,
            default=num_features[: min(3, len(num_features))],
            max_selections=5,
        )

        if selected:
            df_anom = load_sample(dataset_id, [f.lower() for f in selected], sample_size)

            if df_anom.empty:
                st.warning("Selected features not available.")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_anom)

                method = st.selectbox("Anomaly Method", ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"])

                if method == "Isolation Forest":
                    contamination = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01)
                    model = IsolationForest(contamination=contamination, random_state=random_state)
                elif method == "One-Class SVM":
                    nu = st.slider("Nu", 0.01, 0.5, 0.1, 0.01)
                    model = OneClassSVM(nu=nu)
                elif method == "Local Outlier Factor":
                    n_neighbors = st.slider("Neighbors", 5, 50, 20)
                    contamination = st.slider("Contamination", 0.01, 0.5, 0.1, 0.01)
                    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

                if st.button("Detect Anomalies"):
                    with st.spinner("Detecting anomalies..."):
                        if method == "Local Outlier Factor":
                            labels = model.fit_predict(X_scaled)
                        else:
                            labels = model.fit_predict(X_scaled)

                        df_anom["anomaly"] = labels
                        anomaly_count = (labels == -1).sum()
                        st.metric("Anomalies Detected", anomaly_count)

                        if len(selected) == 2:
                            fig = px.scatter(
                                df_anom,
                                x=selected[0].lower(),
                                y=selected[1].lower(),
                                color="anomaly",
                                title=f"Anomaly Detection ({method})",
                                color_discrete_map={1: "blue", -1: "red"}
                            )
                        else:
                            pca = PCA(n_components=2, random_state=random_state)
                            X_pca = pca.fit_transform(X_scaled)
                            df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
                            df_pca["anomaly"] = labels
                            fig = px.scatter(
                                df_pca,
                                x="PC1",
                                y="PC2",
                                color="anomaly",
                                title=f"Anomaly Detection ({method}) - PCA Projection",
                                color_discrete_map={1: "blue", -1: "red"}
                            )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric features available.")

# =====================================================
# TAB 5: OUTLIER ANALYSIS
# =====================================================
with tabs[4]:
    st.markdown("## 📦 Outlier Analysis")

    num_features = [
        f for f, v in eda["summary"].items()
        if v is not None and isinstance(v.get("mean"), (int, float))
    ]

    if num_features:
        feature = st.selectbox("Select numeric feature", num_features)
        method = st.selectbox("Outlier Method", ["Boxplot", "Z-Score", "IQR", "Mahalanobis"])

        df_out = load_sample(dataset_id, [feature.lower()], sample_size)

        if df_out.empty or feature.lower() not in df_out.columns:
            st.warning(f"Feature '{feature}' not available.")
        else:
            data = df_out[feature.lower()].dropna()

            if method == "Boxplot":
                fig = px.box(
                    df_out,
                    y=feature.lower(),
                    points="outliers",
                    title=f"Boxplot – {feature}",
                )
                st.plotly_chart(fig, use_container_width=True)

            elif method == "Z-Score":
                threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
                z_scores = zscore(data)
                outliers = np.abs(z_scores) > threshold
                st.metric("Outliers Detected", outliers.sum())
                fig = px.histogram(
                    x=data,
                    title=f"Z-Score Outliers – {feature}",
                    color=outliers,
                    color_discrete_map={False: "blue", True: "red"}
                )
                st.plotly_chart(fig, use_container_width=True)

            elif method == "IQR":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data < lower_bound) | (data > upper_bound)
                st.metric("Outliers Detected", outliers.sum())
                fig = px.box(
                    y=data,
                    title=f"IQR Outliers – {feature}",
                    points="all"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif method == "Mahalanobis":
                if len(num_features) >= 2:
                    selected_multi = st.multiselect(
                        "Select additional features for Mahalanobis",
                        [f for f in num_features if f != feature],
                        default=num_features[:1] if len(num_features) > 1 else [],
                        max_selections=4
                    )
                    if selected_multi:
                        df_maha = load_sample(dataset_id, [f.lower() for f in [feature] + selected_multi], sample_size)
                        X = df_maha.dropna().values
                        mean = np.mean(X, axis=0)
                        cov = np.cov(X.T)
                        inv_cov = np.linalg.inv(cov)
                        mahal_dist = [mahalanobis(x, mean, inv_cov) for x in X]
                        threshold = st.slider("Mahalanobis Threshold", 1.0, 10.0, 3.0, 0.1)
                        outliers = np.array(mahal_dist) > threshold
                        st.metric("Outliers Detected", outliers.sum())
                        fig = px.scatter(
                            x=X[:, 0],
                            y=X[:, 1] if X.shape[1] > 1 else mahal_dist,
                            color=outliers,
                            title=f"Mahalanobis Outliers – {feature}",
                            color_discrete_map={False: "blue", True: "red"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Select additional features.")
                else:
                    st.info("Need more features for Mahalanobis.")
    else:
        st.info("No numeric features.")

# =====================================================
# TAB 6: FEATURE ANALYSIS
# =====================================================
with tabs[5]:
    st.markdown("## 🔗 Feature Analysis")

    df_feat = load_sample(dataset_id, None, sample_size)

    if not df_feat.empty:
        numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            corr = df_feat[numeric_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature distributions
            feature_dist = st.selectbox("Select feature for distribution", numeric_cols)
            fig_dist = px.histogram(
                df_feat,
                x=feature_dist,
                title=f"Distribution of {feature_dist}"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Scatter plot
            x_feat = st.selectbox("X-axis", numeric_cols, index=0)
            y_feat = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            fig_scatter = px.scatter(
                df_feat,
                x=x_feat,
                y=y_feat,
                title=f"{x_feat} vs {y_feat}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric features for correlation.")
    else:
        st.warning("No data available.")

# =====================================================
# TAB 7: EXPORTS
with tabs[6]:
    st.markdown("## 📥 Executive Reports")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    json_report = json.dumps(
        {
            "eda": eda,
            "etl": etl or "Not run",
            "model": model_meta or "Not trained",
            "generated_at": datetime.now().isoformat(),
        },
        indent=2,
        default=str,
    )

    if export_html:
        html = f"""
        <!DOCTYPE html>
        <html><body><h1>Enterprise EDA Report – {dataset_id}</h1>
        <pre>{json_report}</pre>
        </body></html>
        """
        st.download_button(
            "Download HTML Report",
            html,
            f"EDA_Report_{dataset_id}_{timestamp}.html",
            "text/html",
        )

    if export_pdf:
        st.info("PDF export requires server-side rendering (e.g., WeasyPrint). Source provided.")
        st.download_button(
            "Download JSON (for PDF conversion)",
            json_report,
            f"EDA_Report_{dataset_id}_{timestamp}.json",
            "application/json",
        )

    st.download_button(
        "Download Full JSON Report",
        json_report,
        f"EDA_Report_{dataset_id}_{timestamp}.json",
        "application/json",
    )

st.caption("Enterprise EDA • Unsupervised Intelligence • Audit-Ready • Null-Safe")