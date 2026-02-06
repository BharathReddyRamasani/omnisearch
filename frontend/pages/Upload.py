import streamlit as st
import requests
import pandas as pd

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="OmniSearch AI – Industrial Upload",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# INDUSTRIAL CSS STYLING
# =====================================================
st.markdown("""
<style>
    .upload-header {
        background: linear-gradient(135deg, #0052cc 0%, #1e6ed4 50%, #2563eb 100%);
        padding: 3.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(5, 82, 204, 0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .upload-title {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .upload-subtitle {
        font-size: 1.4rem;
        margin: 1.2rem 0 0 0;
        opacity: 0.95;
        font-weight: 500;
    }
    .upload-zone {
        background: white;
        padding: 3.5rem;
        border-radius: 20px;
        border: 3px dashed #0052cc;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .upload-zone:hover {
        border-color: #2563eb;
        box-shadow: 0 15px 40px rgba(5, 82, 204, 0.15);
        transform: translateY(-4px);
    }
    .upload-icon {
        font-size: 5rem;
        color: #0052cc;
        margin-bottom: 1rem;
    }
    .upload-text {
        font-size: 1.3rem;
        color: #1f2937;
        margin: 1rem 0;
        font-weight: 600;
    }
    .upload-subtext {
        font-size: 1rem;
        color: #6b7280;
        margin: 0.5rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #0052cc, #2563eb);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(5, 82, 204, 0.2);
        margin: 2rem 0;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .success-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .success-title {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .dataset-info {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border-left: 6px solid #1e3c72;
    }
    .info-title {
        color: #1e3c72;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .info-metric {
        display: inline-block;
        background: #e9ecef;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.5rem;
        font-weight: 600;
        color: #495057;
    }
    .next-steps {
        background: linear-gradient(135deg, #0052cc 0%, #2563eb 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 15px 35px rgba(5, 82, 204, 0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .next-steps h3 {
        margin-top: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .step-item {
        background: rgba(255,255,255,0.12);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid rgba(255,255,255,0.4);
        text-align: left;
    }
    .step-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    .error-card {
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        margin: 2rem 0;
    }
    .spinner-container {
        text-align: center;
        padding: 3rem;
    }
    .spinner-text {
        font-size: 1.2rem;
        color: #1e3c72;
        margin-top: 1rem;
    }
    .stFileUploader {
        background: transparent !important;
    }
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    .uploadedFile {
        background: #f8f9fa !important;
        border: 2px solid #1e3c72 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# INDUSTRIAL HEADER
# =====================================================
st.markdown("""
<div class="upload-header">
    <h1 class="upload-title">📤 Industrial Upload</h1>
    <p class="upload-subtitle">Enterprise Dataset Ingestion • Secure Processing • AI-Ready Validation</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =====================================================
# UPLOAD ZONE
# =====================================================
st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📊</div>
    <div class="upload-text"><strong>Drag & Drop or Click to Upload</strong></div>
    <div class="upload-subtext">Supported formats: CSV files only • Maximum size: 100MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="Select a CSV file to upload",
    label_visibility="hidden"
)

# =====================================================
# PROCESSING & RESULTS
# =====================================================
if uploaded_file is not None:
    # Processing Animation
    with st.container():
        st.markdown("""
        <div class="spinner-container">
            <div class="spinner-text">🔄 Processing dataset...</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Processing dataset..."):
            resp = requests.post(
                f"{API}/upload",
                files={"file": uploaded_file}
            )

    if resp.status_code != 200:
        st.markdown(f"""
        <div class="error-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
            <h2>Upload Failed</h2>
            <p>Status Code: {resp.status_code}</p>
            <details>
                <summary>Error Details</summary>
                <pre>{resp.text}</pre>
            </details>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    data = resp.json()

    # Validation
    if "dataset_id" not in data or "columns" not in data:
        st.markdown("""
        <div class="error-card">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
            <h2>Invalid Response</h2>
            <p>The server returned an unexpected response format.</p>
        </div>
        """, unsafe_allow_html=True)
        st.json(data)
        st.stop()

    # Session State Update
    st.session_state.dataset_id = data["dataset_id"]
    st.session_state.columns = data["columns"]
    st.session_state.pop("model_meta", None)
    st.session_state.pop("eda_data", None)

    # Success Card
    st.markdown(f"""
    <div class="success-card">
        <div class="success-icon">✅</div>
        <div class="success-title">Upload Successful!</div>
        <p>Dataset processed and validated successfully</p>
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

    # =====================================================
    # ENCODING DETECTION DETAILS
    # =====================================================
    if "encoding" in data:
        encoding_info = data["encoding"]
        confidence = encoding_info.get("confidence", 0)
        method = encoding_info.get("detection_method", "unknown")
        
        st.markdown("### 🔤 Encoding Detection Report")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detected Encoding", encoding_info.get("detected", "utf-8"))
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        with col3:
            st.metric("Detection Method", method)
        
        st.info(
            f"✓ File detected as **{encoding_info.get('detected')}** with {confidence*100:.1f}% confidence using **{method}**"
        )

    # =====================================================
    # COLUMN NORMALIZATION MAPPING
    # =====================================================
    if "column_mapping" in data and data["column_mapping"]:
        st.markdown("### 🔄 Column Name Normalization")
        st.info("Your column names have been cleaned and normalized to industrial standard format")
        
        # Show mapping table
        mapping = data["column_mapping"]
        mapping_df = pd.DataFrame([
            {"Original": orig, "Normalized": norm}
            for orig, norm in mapping.items()
        ])
        
        st.dataframe(mapping_df, use_container_width=True)
        
        # Confirmation button
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            confirm = st.button("✅ Confirm & Proceed", key="confirm_mapping")
        with col_cancel:
            cancel = st.button("❌ Cancel", key="cancel_mapping")
        
        if cancel:
            st.warning("Upload cancelled. Please re-upload if you want to continue.")
            st.stop()
        
        if not confirm:
            st.info("👆 Please review column mappings and confirm to proceed")
            st.stop()
        
        # Call backend to record confirmation (CRITICAL for data governance)
        try:
            confirm_resp = requests.post(
                f"{API}/datasets/{data['dataset_id']}/confirm-mapping"
            )
            if confirm_resp.status_code == 200:
                st.success("✓ Column mapping confirmed and recorded in audit trail!")
            else:
                st.error(
                    f"❌ **Critical Error**: Failed to record mapping confirmation (HTTP {confirm_resp.status_code}). "
                    f"This is required for data governance. Please try again."
                )
                st.json(confirm_resp.json())
                st.stop()
        except Exception as e:
            st.error(
                f"❌ **Critical Error**: Could not record confirmation: {str(e)}. "
                f"Mapping confirmation is required. Please try again."
            )
            st.stop()
    
    # =====================================================
    # SAMPLE INFO
    # =====================================================
    if "is_sampled" in data and data["is_sampled"]:
        st.warning(
            f"📊 **Dataset is sampled**: Read first {data['sample_limit']:,} rows for processing. "
            f"This is intentional to prevent memory issues with very large files."
        )
    
    # =====================================================
    # TYPE INFERENCE & COERCION SUMMARY
    # =====================================================
    if "coercion_summary" in data:
        coercion_data = data["coercion_summary"]
        
        # Only show columns with coercions
        coerced_cols = {col: info for col, info in coercion_data.items() if info.get("coercions", 0) > 0}
        
        if coerced_cols:
            st.markdown("### 🔧 Type Inference & Automatic Coercions")
            st.info(f"{len(coerced_cols)} columns had types inferred and values automatically coerced")
            
            coercion_df = pd.DataFrame([
                {
                    "Column": col,
                    "Inferred Type": info.get("type", "object"),
                    "Coercions Applied": info.get("coercions", 0),
                    "Method": info.get("method", "none")
                }
                for col, info in coerced_cols.items()
            ])
            
            st.dataframe(coercion_df, use_container_width=True)
    
    # =====================================================
    # DATA PREVIEW
    # =====================================================
    st.markdown("### 🔍 Data Preview")
    st.dataframe(data["preview"], use_container_width=True)

    # Next Steps
    st.markdown("""
    <div class="next-steps">
        <h3>🚀 Next Steps</h3>
        <div class="step-item">
            <span class="step-icon">📊</span>
            <strong>EDA (Exploratory Data Analysis):</strong> Analyze distributions, correlations, and data quality
        </div>
        <div class="step-item">
            <span class="step-icon">🧹</span>
            <strong>ETL (Extract, Transform, Load):</strong> Clean data, handle missing values, feature engineering
        </div>
        <div class="step-item">
            <span class="step-icon">🤖</span>
            <strong>Train:</strong> AutoML training with 10+ algorithms and advanced metrics
        </div>
        <div class="step-item">
            <span class="step-icon">🔮</span>
            <strong>Predict:</strong> Deploy models and make real-time predictions
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Empty State
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #6c757d;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">📤</div>
        <h3>Ready to Upload</h3>
        <p>Select a CSV file above to begin the industrial ML pipeline</p>
    </div>
    """, unsafe_allow_html=True)
