import streamlit as st

st.set_page_config(
    page_title="OmniSearch AI - Industrial ML Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

from theme import inject_theme, page_header, page_footer
inject_theme()

# Initialize session state
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "columns" not in st.session_state:
    st.session_state.columns = []

# Main header
page_header("🚀", "OmniSearch AI", "Industrial Machine Learning Platform • Enterprise-Grade AutoML")

# Workflow pipeline steps
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="step-card">
        <span class="step-icon">📤</span>
        <div class="step-title">Upload</div>
        <div class="step-desc">Load datasets</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="step-card">
        <span class="step-icon">🔍</span>
        <div class="step-title">EDA</div>
        <div class="step-desc">Explore & analyze</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="step-card">
        <span class="step-icon">⚙️</span>
        <div class="step-title">ETL</div>
        <div class="step-desc">Clean & transform</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="step-card">
        <span class="step-icon">🤖</span>
        <div class="step-title">Train</div>
        <div class="step-desc">AutoML training</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="step-card">
        <span class="step-icon">🔮</span>
        <div class="step-title">Predict</div>
        <div class="step-desc">Deploy & predict</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Current status
st.markdown("### 📊 Current Session")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    dataset_status = "✅ Loaded" if st.session_state.dataset_id else "⏳ Not loaded"
    st.metric("Dataset", dataset_status)

with status_col2:
    columns_count = len(st.session_state.columns) if st.session_state.columns else 0
    st.metric("Columns", columns_count)

with status_col3:
    model_status = "✅ Trained" if st.session_state.get("model_meta") else "⏳ Not trained"
    st.metric("Model", model_status)

# Getting started
st.markdown("""
<div class="info-card">
    <h4>🚀 Getting Started</h4>
    <p>Use the sidebar to follow the ML workflow:</p>
    <ol>
        <li><strong>Upload</strong> — Load your dataset (CSV)</li>
        <li><strong>EDA</strong> — Explore distributions, correlations, and patterns</li>
        <li><strong>ETL</strong> — Clean data, handle missing values, feature engineering</li>
        <li><strong>Train</strong> — Train ML models with automatic comparison</li>
        <li><strong>Predict</strong> — Make predictions using trained models</li>
    </ol>
    <p><strong>Features:</strong> 10+ algorithms, advanced metrics, confusion matrices, feature importance, batch processing.</p>
</div>
""", unsafe_allow_html=True)

page_footer()
