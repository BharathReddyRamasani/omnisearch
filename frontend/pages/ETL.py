# import streamlit as st
# import requests

# API = "http://127.0.0.1:8000/api"

# st.set_page_config(
#     layout="wide",
#     page_title="OmniSearch AI — Enterprise ETL"
# )

# st.markdown("## 🧹 Enterprise ETL Pipeline")

# # ---------------- DATASET VALIDATION ----------------
# dataset_id = st.session_state.get("dataset_id")

# if not dataset_id:
#     st.error("No dataset loaded. Upload a dataset first.")
#     st.stop()

# st.success(f"Active Dataset: `{dataset_id}`")

# # ---------------- ETL EXECUTION ----------------
# col1, col2 = st.columns([3, 1])

# with col1:
#     if st.button("🚀 Execute Enterprise ETL", type="primary", use_container_width=True):
#         with st.spinner("Running ETL (dedup • imputation • outliers • quality scoring)…"):
#             r = requests.post(f"{API}/datasets/{dataset_id}/clean")

#             if r.status_code == 200:
#                 st.session_state.etl_done = True
#                 st.success("ETL completed successfully")
#                 st.rerun()
#             else:
#                 if r.headers.get("content-type", "").startswith("application/json"):
#                     st.error(r.json().get("detail", "ETL failed"))
#                 else:
#                     st.error("ETL failed — non-JSON backend response")
#                     st.code(r.text)

# with col2:
#     if st.button("📊 Load Comparison", use_container_width=True):
#         st.session_state.show_comp = True
#         st.rerun()

# # ---------------- DOWNLOAD ZONE ----------------
# st.markdown("---")
# st.markdown("## ⬇️ Dataset Artifacts")

# d1, d2 = st.columns(2)

# # CLEAN
# with d1:
#     r = requests.get(f"{API}/datasets/{dataset_id}/download/clean")
#     if r.status_code == 200:
#         st.download_button(
#             "Download Clean Dataset",
#             data=r.content,
#             file_name=f"{dataset_id}_clean.csv",
#             mime="text/csv",
#             use_container_width=True
#         )
#     else:
#         st.info("Clean dataset not available (ETL not run yet)")

# # RAW
# with d2:
#     r = requests.get(f"{API}/datasets/{dataset_id}/download/raw")
#     if r.status_code == 200:
#         st.download_button(
#             "Download Raw Dataset",
#             data=r.content,
#             file_name=f"{dataset_id}_raw.csv",
#             mime="text/csv",
#             use_container_width=True
#         )

# # ---------------- COMPARISON ----------------
# st.markdown("---")
# st.markdown("## 📊 ETL Impact Report")

# if st.session_state.get("show_comp") or st.session_state.get("etl_done"):
#     r = requests.get(f"{API}/datasets/{dataset_id}/comparison")

#     if r.status_code != 200:
#         st.warning("Run ETL to generate comparison report")
#         st.stop()

#     comp = r.json()

#     c1, c2, c3, c4 = st.columns(4)

#     c1.metric("Rows", comp["raw_stats"]["rows"], comp["clean_stats"]["rows"])
#     c2.metric("Missing Filled", comp["improvements"]["missing_values_filled"])
#     c3.metric("Outliers Fixed", comp["improvements"]["outliers_fixed"])
#     c4.metric("Expected Lift", f"+{comp['accuracy_lift_expected']}%")

#     st.markdown("### Detailed ETL Intelligence")
#     st.json(comp)

# # ---------------- NEXT STEPS ----------------
# st.markdown("---")
# st.markdown("## 🚀 Production Flow")

# n1, n2, n3 = st.columns(3)

# with n1:
#     if st.button("📈 EDA (Clean)", use_container_width=True):
#         st.switch_page("pages/EDA.py")


# with n2:
#     if st.button("Train Model", use_container_width=True):
#         st.switch_page("pages/Train.py")
# with n3:
#     if st.button("Predict", use_container_width=True):
#         st.switch_page("pages/Predict.py")

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

API = "http://127.0.0.1:8003/api"

st.set_page_config(layout="wide", page_title="OmniSearch AI — Enterprise ETL")

st.markdown("## 🧹 Enterprise ETL Pipeline")

dataset_id = st.session_state.get("dataset_id")
if not dataset_id:
    st.error("No dataset loaded. Upload a dataset first.")
    st.stop()

st.success(f"Active Dataset: `{dataset_id}`")

# =====================================================
# ETL TABS
# =====================================================
etl_tabs = st.tabs(["⚙️ Configuration", "🚀 Execution", "📊 Results", "🔧 Advanced"])

# =====================================================
# TAB 1: CONFIGURATION
# =====================================================
with etl_tabs[0]:
    st.markdown("### ETL Pipeline Configuration")

    st.markdown("**Standard ETL Steps (Backend-Automated):**")
    st.checkbox("Remove Duplicates", value=True, disabled=True)
    st.checkbox("Handle Missing Values (Imputation)", value=True, disabled=True)
    st.checkbox("Outlier Detection & Fixing", value=True, disabled=True)
    st.checkbox("Data Type Standardization", value=True, disabled=True)
    st.checkbox("Quality Scoring", value=True, disabled=True)

    st.markdown("---")
    st.markdown("**Imputation Methods (Applied by Backend):**")
    imputation = st.selectbox("Primary Imputation", ["Mean", "Median", "KNN", "Forward Fill"], index=1, disabled=True)
    outlier_method = st.selectbox("Outlier Handling", ["IQR", "Z-Score", "Isolation Forest"], index=0, disabled=True)

    st.info("Configuration is handled automatically by the backend pipeline for optimal results.")

# =====================================================
# TAB 2: EXECUTION
# =====================================================
with etl_tabs[1]:
    st.markdown("### Execute ETL Pipeline")

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("🚀 Execute Enterprise ETL", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing ETL pipeline...")
            progress_bar.progress(10)

            status_text.text("Removing duplicates...")
            progress_bar.progress(30)

            status_text.text("Handling missing values...")
            progress_bar.progress(50)

            status_text.text("Detecting and fixing outliers...")
            progress_bar.progress(70)

            status_text.text("Finalizing quality scoring...")
            progress_bar.progress(90)

            with st.spinner("Running ETL…"):
                r = requests.post(f"{API}/datasets/{dataset_id}/clean")
                if r.status_code != 200:
                    st.error(r.text)
                    st.stop()
                st.session_state.etl_done = True
                progress_bar.progress(100)
                status_text.text("ETL completed successfully!")
                st.success("ETL completed successfully")
                st.rerun()

    with col2:
        st.metric("ETL Status", "Ready" if not st.session_state.get("etl_done") else "Completed")

# =====================================================
# TAB 3: RESULTS
# =====================================================
with etl_tabs[2]:
    st.markdown("### ETL Impact Report & Downloads")

    # Downloads
    st.markdown("## ⬇️ Dataset Artifacts")

    d1, d2 = st.columns(2)

    with d1:
        r = requests.get(f"{API}/datasets/{dataset_id}/download/clean")
        if r.status_code == 200:
            st.download_button(
                "Download Clean Dataset",
                r.content,
                file_name=f"{dataset_id}_clean.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Clean dataset not available (ETL not run yet)")

    with d2:
        r = requests.get(f"{API}/datasets/{dataset_id}/download/raw")
        if r.status_code == 200:
            st.download_button(
                "Download Raw Dataset",
                r.content,
                file_name=f"{dataset_id}_raw.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Comparison
    st.markdown("---")
    st.markdown("## 📊 ETL Impact Report")

    r = requests.get(f"{API}/datasets/{dataset_id}/comparison")
    if r.status_code == 200:
        comp = r.json()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", comp["raw_stats"]["rows"], comp["clean_stats"]["rows"] - comp["raw_stats"]["rows"])
        c2.metric("Missing Filled", comp["improvements"]["missing_values_filled"])
        c3.metric("Duplicates Removed", comp["improvements"]["duplicates_removed"])
        c4.metric("Outliers Fixed", comp["improvements"].get("outliers_fixed", 0))

        st.markdown("### Detailed ETL Intelligence")
        st.json(comp)
    else:
        st.info("Run ETL to generate comparison report")

# =====================================================
# TAB 4: ADVANCED
# =====================================================
with etl_tabs[3]:
    st.markdown("### Advanced ETL Options & Manual Fixes")

    st.markdown("**Client-Side Outlier Detection (Preview):**")
    # Load sample for preview
    try:
        r = requests.get(f"{API}/datasets/{dataset_id}/download/raw", timeout=30)
        if r.status_code == 200:
            df_preview = pd.read_csv(pd.io.common.StringIO(r.text))
            numeric_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()

            if numeric_cols:
                selected_col = st.selectbox("Select column for outlier preview", numeric_cols)
                method = st.selectbox("Detection Method", ["IQR", "Z-Score", "Isolation Forest"])

                data = df_preview[selected_col].dropna()

                if method == "IQR":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = (data < lower) | (data > upper)
                elif method == "Z-Score":
                    z_scores = (data - data.mean()) / data.std()
                    outliers = np.abs(z_scores) > 3
                elif method == "Isolation Forest":
                    iso = IsolationForest(contamination=0.1, random_state=42)
                    outliers = iso.fit_predict(data.values.reshape(-1, 1)) == -1

                st.metric("Outliers Detected", outliers.sum())
                st.metric("Total Values", len(data))

                # Plot
                fig = px.box(df_preview, y=selected_col, points="all", title=f"Outlier Preview - {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for outlier detection.")
        else:
            st.info("Raw data not available.")
    except Exception as e:
        st.error(f"Error loading preview: {str(e)}")

    st.markdown("---")
    st.markdown("**Next Steps:**")
    n1, n2, n3 = st.columns(3)

    with n1:
        if st.button("📈 EDA", use_container_width=True):
            st.switch_page("pages/EDA.py")
    with n2:
        if st.button("Train Model", use_container_width=True):
            st.switch_page("pages/Train.py")
    with n3:
        if st.button("Predict", use_container_width=True):
            st.switch_page("pages/Predict.py")
