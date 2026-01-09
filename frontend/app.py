import streamlit as st

st.set_page_config(
    page_title="OmniSearch AI - Industrial ML Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚀"
)

# Custom CSS for industrial look
st.markdown("""
<style>
    /* Global overrides for extreme visibility and attractiveness */
    body {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stApp {
        background-color: #ffffff !important;
    }
    .css-1d391kg, .css-12oz5g7, .css-1r6slb0 {
        background-color: #ffffff !important;
    }
    /* Ensure all text is visible */
    .stText, .stMarkdown, .stHeader, .stSubheader, p, h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
    /* Sidebar styling */
    .css-1lcbmhc, .css-1outpf7 {
        background: linear-gradient(180deg, #34495e, #2c3e50) !important;
        color: white !important;
    }
    .css-1lcbmhc .stRadio label, .css-1outpf7 .stRadio label {
        color: white !important;
    }
    /* Main content area */
    .css-1y4p8pa {
        background-color: #f8f9fa !important;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72, #2a5298, #3a7bd5, #4a90e2);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
        border: 2px solid rgba(255,255,255,0.1);
    }
    .main-title {
        font-size: 4rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        letter-spacing: 2px;
    }
    .main-subtitle {
        font-size: 1.5rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 300;
    }
    .workflow-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 2rem;
        border-radius: 15px;
        border-left: 6px solid #1e3c72;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
    }
    .workflow-card h4 {
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    .workflow-card ol li {
        margin: 0.5rem 0;
        color: #495057;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        border: 2px solid rgba(255,255,255,0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    }
    .metric-card h4 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: bold;
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
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #e9ecef;
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
