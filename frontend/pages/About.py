import streamlit as st

from theme import inject_theme, page_header, page_footer
inject_theme()

page_header("ℹ️", "About OmniSearch AI", "Industrial Machine Learning Platform")

st.markdown("""
<div class="info-card">
    <h4>🚀 OmniSearch AI</h4>
    <p>A production-grade CSV intelligence platform for end-to-end machine learning workflows.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#B8B3FF; margin:0 0 0.75rem 0;">✨ Features</h4>
        <ul style="color:#9AA0A6; line-height:1.8;">
            <li>Dirty CSV handling with encoding detection</li>
            <li>Enterprise EDA with interactive plots</li>
            <li>AutoML training with 10+ algorithms</li>
            <li>Real-time prediction with confidence scoring</li>
            <li>Batch prediction & export</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#B8B3FF; margin:0 0 0.75rem 0;">🛠️ Tech Stack</h4>
        <ul style="color:#9AA0A6; line-height:1.8;">
            <li><strong style="color:#E8EAED;">Frontend:</strong> Streamlit</li>
            <li><strong style="color:#E8EAED;">Backend:</strong> FastAPI</li>
            <li><strong style="color:#E8EAED;">ML:</strong> Scikit-learn, Pandas, NumPy</li>
            <li><strong style="color:#E8EAED;">Serialization:</strong> Joblib</li>
            <li><strong style="color:#E8EAED;">Viz:</strong> Plotly</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-card" style="margin-top:1.5rem;">
    <h4>📋 ML Pipeline</h4>
    <p>Upload → EDA → ETL → Train → Predict — a complete end-to-end machine learning workflow with enterprise-grade features.</p>
</div>
""", unsafe_allow_html=True)

page_footer()
