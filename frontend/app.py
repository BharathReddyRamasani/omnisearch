import streamlit as st

st.set_page_config(
    page_title="OmniSearch AI - Industrial ML Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS for modern blue gradient design
st.markdown("""
<style>
    /* Global theme colors: Modern blue gradients */
    body {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }
    .stApp {
        background-color: #f9fafb !important;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0052cc 0%, #1e6ed4 50%, #2563eb 100%);
        padding: 3.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 40px rgba(5, 82, 204, 0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .main-subtitle {
        font-size: 1.4rem;
        margin: 1.2rem 0 0 0;
        opacity: 0.95;
        font-weight: 500;
    }
    
    /* Workflow cards */
    .workflow-card {
        background: linear-gradient(135deg, #f0f6ff 0%, #e6f2ff 100%);
        padding: 2rem;
        border-radius: 16px;
        border-left: 5px solid #0052cc;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #dbeafe;
        transition: all 0.3s ease;
        color: #1f2937;
    }
    .workflow-card:hover {
        box-shadow: 0 8px 30px rgba(5, 82, 204, 0.1);
        transform: translateY(-2px);
    }
    .workflow-card h4 {
        color: #0052cc;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .workflow-card ol li {
        margin: 0.75rem 0;
        color: #4b5563;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #0052cc 0%, #2563eb 100%);
        color: white;
        padding: 1.75rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(5, 82, 204, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(5, 82, 204, 0.25);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .sidebar-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .sidebar-header h3 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: bold;
    }
    /* Status metrics */
    .stMetric {
        background: linear-gradient(135deg, #f0f6ff 0%, #e6f2ff 100%);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #dbeafe;
        color: #1f2937;
    }
    .stMetric label {
        color: #6c757d !important;
        font-weight: 600;
    }
    .stMetric .metric-value {
        color: #1e3c72 !important;
        font-size: 1.5rem;
        font-weight: bold;
    }
    /* Footer */
    .stCaption {
        text-align: center;
        color: #6c757d !important;
        font-style: italic;
    }
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #218838, #1aa085);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Links */
    a {
        color: #1e3c72 !important;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "columns" not in st.session_state:
    st.session_state.columns = []

# Main header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🚀 OmniSearch AI</h1>
    <p class="main-subtitle">Industrial Machine Learning Platform • Enterprise-Grade AutoML</p>
</div>
""", unsafe_allow_html=True)

# Workflow overview
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>📤 Upload</h4>
        <p>Load your datasets</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🔍 EDA</h4>
        <p>Explore & analyze</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>⚙️ ETL</h4>
        <p>Clean & transform</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4>🤖 Train</h4>
        <p>AutoML training</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <h4>🔮 Predict</h4>
        <p>Deploy & predict</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Current status
st.markdown("### 📊 Current Session Status")

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

# Instructions
st.markdown("""
<div class="workflow-card">
    <h4>🚀 Getting Started</h4>
    <p>Use the sidebar navigation to follow the ML workflow:</p>
    <ol>
        <li><strong>Upload:</strong> Load your dataset (CSV, Excel, etc.)</li>
        <li><strong>EDA:</strong> Explore data distributions, correlations, and patterns</li>
        <li><strong>ETL:</strong> Clean data, handle missing values, and feature engineering</li>
        <li><strong>Train:</strong> Train multiple ML models with automatic comparison</li>
        <li><strong>Predict:</strong> Make predictions on new data using trained models</li>
    </ol>
    <p><strong>Features:</strong> 10+ algorithms, advanced metrics, confusion matrices, feature importance, batch processing, and enterprise-grade exports.</p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("© 2025 OmniSearch AI • Industrial ML Platform • Enterprise AutoML • Built with FastAPI & Streamlit")
