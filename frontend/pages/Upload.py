import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8001/api"

st.set_page_config(
    page_title="OmniSearch AI – Upload",
    layout="wide",
    initial_sidebar_state="expanded"
)

from theme import inject_theme, page_header, page_footer
inject_theme()

# Header
page_header("📤", "Upload Dataset", "Enterprise Dataset Ingestion • AI-Ready Validation")

st.markdown("---")

# Upload zone
st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📊</div>
    <div class="upload-text">Drag & Drop or Click to Upload</div>
    <div class="upload-subtext">Supported: CSV files • Max size: 100MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Select a CSV file to upload",
    label_visibility="hidden"
)

if uploaded_file is not None:
    with st.spinner("🔄 Processing dataset..."):
        resp = requests.post(
            f"{API}/upload",
            files={"file": uploaded_file}
        )

    if resp.status_code != 200:
        st.markdown(f"""
        <div class="error-card">
            <div style="font-size: 2.5rem;">❌</div>
            <h3>Upload Failed</h3>
            <p>Status Code: {resp.status_code}</p>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Error Details"):
            st.code(resp.text)
        st.stop()

    data = resp.json()

    if "dataset_id" not in data or "columns" not in data:
        st.markdown("""
        <div class="error-card">
            <div style="font-size: 2.5rem;">⚠️</div>
            <h3>Invalid Response</h3>
            <p>The server returned an unexpected format.</p>
        </div>
        """, unsafe_allow_html=True)
        st.json(data)
        st.stop()

    # Session State
    st.session_state.dataset_id = data["dataset_id"]
    st.session_state.columns = data["columns"]
    st.session_state.pop("model_meta", None)
    st.session_state.pop("eda_data", None)

    # Success
    st.markdown("""
    <div class="success-card">
        <div class="success-icon">✅</div>
        <h3>Upload Successful!</h3>
        <p>Dataset processed and validated</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset Info
    st.markdown(f"""
    <div class="dataset-info">
        <div class="info-title">📊 Dataset Information</div>
        <div class="info-metric">ID: {data['dataset_id']}</div>
        <div class="info-metric">Columns: {len(data['columns'])}</div>
        <div class="info-metric">Rows: {len(data['preview'])}</div>
    </div>
    """, unsafe_allow_html=True)

    # Encoding detection
    if "encoding" in data:
        encoding_info = data["encoding"]
        confidence = encoding_info.get("confidence", 0)
        method = encoding_info.get("detection_method", "unknown")
        
        st.markdown("### 🔤 Encoding Detection")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Encoding", encoding_info.get("detected", "utf-8"))
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Method", method)

    # Column mapping
    if "column_mapping" in data and data["column_mapping"]:
        st.markdown("### 🔄 Column Normalization")
        st.info("Column names cleaned to standard format")
        
        mapping = data["column_mapping"]
        mapping_df = pd.DataFrame([
            {"Original": orig, "Normalized": norm}
            for orig, norm in mapping.items()
        ])
        st.dataframe(mapping_df, use_container_width=True)
        
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            confirm = st.button("✅ Confirm & Proceed", key="confirm_mapping")
        with col_cancel:
            cancel = st.button("❌ Cancel", key="cancel_mapping")
        
        if cancel:
            st.warning("Upload cancelled.")
            st.stop()
        
        if not confirm:
            st.info("👆 Review and confirm column mappings")
            st.stop()
        
        try:
            confirm_resp = requests.post(f"{API}/datasets/{data['dataset_id']}/confirm-mapping")
            if confirm_resp.status_code == 200:
                st.success("✓ Column mapping confirmed!")
            else:
                st.error(f"Failed to confirm mapping (HTTP {confirm_resp.status_code})")
                st.stop()
        except Exception as e:
            st.error(f"Could not record confirmation: {str(e)}")
            st.stop()

    # Sampling info
    if "is_sampled" in data and data["is_sampled"]:
        st.warning(f"📊 Dataset sampled to {data['sample_limit']:,} rows for processing.")

    # Coercion summary
    if "coercion_summary" in data:
        coercion_data = data["coercion_summary"]
        coerced_cols = {col: info for col, info in coercion_data.items() if info.get("coercions", 0) > 0}
        
        if coerced_cols:
            st.markdown("### 🔧 Type Coercions")
            coercion_df = pd.DataFrame([
                {
                    "Column": col,
                    "Inferred Type": info.get("type", "object"),
                    "Coercions": info.get("coercions", 0),
                    "Method": info.get("method", "none")
                }
                for col, info in coerced_cols.items()
            ])
            st.dataframe(coercion_df, use_container_width=True)

    # Preview
    st.markdown("### 🔍 Data Preview")
    st.dataframe(data["preview"], use_container_width=True)

    # Next Steps
    st.markdown("""
    <div class="next-steps">
        <h3>🚀 Next Steps</h3>
        <div class="step-item">
            <strong>EDA:</strong> Analyze distributions, correlations, and data quality
        </div>
        <div class="step-item">
            <strong>ETL:</strong> Clean data, handle missing values, feature engineering
        </div>
        <div class="step-item">
            <strong>Train:</strong> AutoML with 10+ algorithms and metrics
        </div>
        <div class="step-item">
            <strong>Predict:</strong> Deploy and make real-time predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #9AA0A6;">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">📤</div>
        <h3 style="color: #E8EAED;">Ready to Upload</h3>
        <p>Select a CSV file above to begin</p>
    </div>
    """, unsafe_allow_html=True)

page_footer()
