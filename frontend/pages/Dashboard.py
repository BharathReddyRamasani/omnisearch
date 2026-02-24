import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json

# Update API URL
API = "http://127.0.0.1:8001/api"

st.set_page_config(
    page_title="OmniSearch AI – Industrial Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

from theme import inject_theme, page_header, page_footer
inject_theme()

# =====================================================
# SIDEBAR METRICS
# =====================================================
with st.sidebar:
    st.markdown("### 📊 **System Status**")
    st.markdown("---")

    # System health indicators
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🖥️ CPU", "45%", "↓ 2%")
    with col2:
        st.metric("💾 Memory", "67%", "↑ 5%")

    st.markdown("---")
    st.markdown("### 🔄 **Active Sessions**")

    sessions = [
        {"user": "Data Scientist", "dataset": "customer_data.csv", "status": "Training"},
        {"user": "ML Engineer", "dataset": "sales_data.csv", "status": "Predicting"},
        {"user": "Analyst", "dataset": "inventory.csv", "status": "EDA"}
    ]

    for session in sessions:
        st.markdown(f"""
        <div class="glass-card" style="padding:0.8rem 1rem;">
            <strong style="color:#E8EAED;">{session['user']}</strong><br>
            <small style="color:#9AA0A6;">{session['dataset']}</small><br>
            <span style="color: #2DD4A0;">● {session['status']}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚨 **Alerts**")

    alerts = [
        {"type": "warning", "message": "High memory usage detected", "time": "2 min ago"},
        {"type": "info", "message": "New model deployed", "time": "5 min ago"}
    ]

    for alert in alerts:
        color = "rgba(240, 180, 41, 0.15)" if alert["type"] == "warning" else "rgba(108, 99, 255, 0.1)"
        border_color = "#F0B429" if alert["type"] == "warning" else "#6C63FF"
        st.markdown(f"""
        <div style="background:{color}; border:1px solid {border_color}; color:#E8EAED; padding:0.5rem 0.8rem; border-radius:8px; margin:0.5rem 0; font-size:0.8rem;">
            {alert['message']}<br><small style="color:#9AA0A6;">{alert['time']}</small>
        </div>
        """, unsafe_allow_html=True)

# =====================================================
# MAIN DASHBOARD HEADER
# =====================================================
page_header("🚀", "Industrial ML Dashboard", "Real-time Analytics • Enterprise Monitoring • AI-Powered Insights")

# =====================================================
# SESSION CHECK
# =====================================================
if "dataset_id" not in st.session_state:
    st.warning("⚠️ **No Dataset Loaded**")
    st.info("Please upload a dataset from the **Upload** page to unlock the full dashboard experience.")
    st.stop()

dataset_id = st.session_state.dataset_id

# =====================================================
# REAL-TIME METRICS ROW
# =====================================================
st.markdown("## 📈 **Real-Time KPIs**")

# Fetch data for metrics
try:
    # Get dataset info
    info_resp = requests.get(f"{API}/datasets/{dataset_id}/info", timeout=5)
    dataset_info = info_resp.json() if info_resp.status_code == 200 else {}

    # Get EDA data - Extract from nested 'eda' key if present
    eda_resp = requests.get(f"{API}/eda/{dataset_id}", timeout=10)
    eda_response = eda_resp.json() if eda_resp.status_code == 200 else {}
    eda_data = eda_response.get('eda', eda_response)  # Handle both nested and flat responses

    # Get model metadata
    meta_resp = requests.get(f"{API}/meta/{dataset_id}", timeout=5)
    model_meta = meta_resp.json() if meta_resp.status_code == 200 else {}

except Exception as e:
    st.error(f"Connection error: {str(e)}")
    dataset_info = {}
    eda_data = {}
    model_meta = {}

# KPI Cards — using step-card style for dark theme
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">📊 Dataset Rows</div>
        <div class="step-title" style="font-size:1.5rem;">{dataset_info.get('rows', 0):,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    cols_data = dataset_info.get('columns', 0)
    cols_count = len(cols_data) if isinstance(cols_data, list) else cols_data
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">📋 Columns</div>
        <div class="step-title" style="font-size:1.5rem;">{cols_count}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    missing_data = eda_data.get('missing', {})
    missing_total = sum(missing_data.values()) if isinstance(missing_data, dict) else 0
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">⚠️ Missing Values</div>
        <div class="step-title" style="font-size:1.5rem;">{missing_total:,}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    model_score = model_meta.get('best_score', 0)
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">🎯 Model Score</div>
        <div class="step-title" style="font-size:1.5rem;">{model_score:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col5:
    models_trained = len(model_meta.get('leaderboard', []))
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">🤖 Models Trained</div>
        <div class="step-title" style="font-size:1.5rem;">{models_trained}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col6:
    uptime = "24h 32m"
    st.markdown(f"""
    <div class="step-card">
        <div class="step-desc">⏱️ System Uptime</div>
        <div class="step-title" style="font-size:1.5rem;">{uptime}</div>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# DASHBOARD TABS
# =====================================================
tab_overview, tab_analytics, tab_models, tab_monitoring = st.tabs([
    "📊 Overview", "📈 Analytics", "🤖 Models", "🔍 Monitoring"
])

# =====================================================
# TAB 1: OVERVIEW
# =====================================================
with tab_overview:
    st.markdown("## 🎯 **Project Overview**")

    # Status cards
    status_col1, status_col2, status_col3 = st.columns(3)

    with status_col1:
        dataset_status = "✅ Active" if dataset_id else "❌ Not Loaded"
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:#B8B3FF; margin:0 0 0.5rem 0;">📁 Dataset Status</h4>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; color:#E8EAED;">{dataset_status}</p>
            <small style="color:#9AA0A6;">ID: {dataset_id[:8]}...</small>
        </div>
        """, unsafe_allow_html=True)

    with status_col2:
        model_status = "✅ Trained" if model_meta else "⏳ Not Trained"
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:#B8B3FF; margin:0 0 0.5rem 0;">🤖 Model Status</h4>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; color:#E8EAED;">{model_status}</p>
            <small style="color:#9AA0A6;">Best: {model_meta.get('best_model', 'N/A')}</small>
        </div>
        """, unsafe_allow_html=True)

    with status_col3:
        pipeline_status = "✅ Complete" if model_meta else "🔄 In Progress"
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color:#B8B3FF; margin:0 0 0.5rem 0;">🔄 Pipeline Status</h4>
            <p style="font-size: 1.2rem; margin: 0.5rem 0; color:#E8EAED;">{pipeline_status}</p>
            <small style="color:#9AA0A6;">AutoML Active</small>
        </div>
        """, unsafe_allow_html=True)

    # Workflow progress
    st.markdown("### 📋 **ML Pipeline Progress**")

    progress_data = {
        "Upload": 100,
        "EDA": 100 if eda_data else 0,
        "ETL": 100 if eda_data else 0,  # Assuming ETL is done if EDA exists
        "Train": 100 if model_meta else 0,
        "Predict": 50 if model_meta else 0
    }

    progress_df = pd.DataFrame(list(progress_data.items()), columns=['Step', 'Progress'])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=progress_df['Step'],
        y=progress_df['Progress'],
        marker_color=['#0052cc', '#0052cc', '#0052cc', '#0052cc', '#2563eb'],
        text=[f"{p}%" for p in progress_df['Progress']],
        textposition='auto'
    ))

    fig.update_layout(
        title="Pipeline Completion Status",
        xaxis_title="Pipeline Steps",
        yaxis_title="Completion (%)",
        height=300,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# TAB 2: ANALYTICS
# =====================================================
with tab_analytics:
    st.markdown("## 📈 **Advanced Analytics**")

    if eda_data:
        # Data quality metrics
        st.markdown("### 🔍 **Data Quality Overview**")

        quality_col1, quality_col2, quality_col3 = st.columns(3)

        with quality_col1:
            # Safely calculate completeness
            missing_data = eda_data.get('missing', {})
            rows = dataset_info.get('rows', 0)
            cols_data = dataset_info.get('columns', 0)
            # Handle columns as either list or int
            cols = len(cols_data) if isinstance(cols_data, list) else cols_data
            
            if rows > 0 and cols > 0 and isinstance(missing_data, dict):
                missing_count = sum(missing_data.values())
                completeness = (1 - (missing_count / (rows * cols))) * 100
            else:
                completeness = 100
            st.metric("Data Completeness", f"{max(0, completeness):.1f}%")

        with quality_col2:
            # Get numeric columns count
            summary = eda_data.get('summary', {})
            numeric_cols = sum(1 for v in summary.values() if v and isinstance(v, dict) and 'mean' in v)
            st.metric("Numeric Features", numeric_cols)

        with quality_col3:
            # Get categorical columns count
            categorical_cols = sum(1 for v in summary.values() if v and isinstance(v, dict) and 'unique' in v)
            st.metric("Categorical Features", categorical_cols)

        # Missing values visualization
        st.markdown("### 📊 **Missing Values Analysis**")

        missing_data = eda_data.get('missing', {})
        if missing_data and isinstance(missing_data, dict):
            missing_df = pd.DataFrame(list(missing_data.items()), columns=['Column', 'Missing Count'])
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

            if not missing_df.empty:
                fig = px.bar(
                    missing_df,
                    x='Column',
                    y='Missing Count',
                    title="Missing Values by Column",
                    color='Missing Count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("🎉 No missing values detected!")

        # Data types distribution from summary statistics
        st.markdown("### 🏷️ **Data Types Distribution**")

        summary = eda_data.get('summary', {})
        if summary:
            numeric_count = sum(1 for v in summary.values() if v and isinstance(v, dict) and 'mean' in v)
            categorical_count = sum(1 for v in summary.values() if v and isinstance(v, dict) and 'unique' in v)
            
            if numeric_count > 0 or categorical_count > 0:
                type_counts = pd.Series({'Numeric': numeric_count, 'Categorical': categorical_count})

                fig = px.pie(
                    values=type_counts.values,
                    names=type_counts.index,
                    title="Data Types Distribution",
                    color_discrete_sequence=['#0052cc', '#2563eb']
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("📊 Run EDA analysis to unlock advanced analytics")

# =====================================================
# TAB 3: MODELS
# =====================================================
with tab_models:
    st.markdown("## 🤖 **Model Performance Dashboard**")

    if model_meta:
        # Model leaderboard
        st.markdown("### 🏆 **Model Leaderboard**")

        leaderboard = model_meta.get('leaderboard', [])
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard)
            
            # ✅ Determine which score column to sort by
            sort_col = None
            if 'ranking_score' in df_lb.columns:
                sort_col = 'ranking_score'
            elif 'holdout_score' in df_lb.columns:
                sort_col = 'holdout_score'
            elif 'score' in df_lb.columns:
                sort_col = 'score'
            
            if sort_col:
                df_lb = df_lb.sort_values(sort_col, ascending=False)
                
                # Enhanced leaderboard with styling
                st.dataframe(
                    df_lb.style
                    .background_gradient(subset=[sort_col], cmap='Blues')
                    .format({sort_col: '{:.4f}'})
                    .apply(lambda x: ['background-color: #e6f7ff' if x.name == 0 else '' for i in x], axis=0),
                    use_container_width=True
                )
            else:
                st.dataframe(df_lb, use_container_width=True)
        else:
            st.info("No models trained yet")

            # Performance radar chart
            st.markdown("### 📈 **Model Comparison Radar**")

            # Create radar chart for top 3 models
            categories = ['Accuracy/Score', 'Train Size', 'Test Size']

            fig = go.Figure()

            for i, row in df_lb.head(3).iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['score'], row['train_rows']/max(df_lb['train_rows']), row['test_rows']/max(df_lb['test_rows'])],
                    theta=categories,
                    fill='toself',
                    name=row['model']
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Model Performance Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Model details
        st.markdown("### 🔍 **Champion Model Details**")

        champion_col1, champion_col2 = st.columns(2)

        with champion_col1:
            st.markdown(f"""
            <div class="step-card" style="text-align:left; padding:1.5rem;">
                <h4 style="color:#B8B3FF; margin:0 0 0.5rem 0;">🏆 Best Model</h4>
                <h2 style="color:#E8EAED; margin:0.5rem 0;">{model_meta.get('best_model', 'N/A')}</h2>
                <p style="color:#9AA0A6;">Score: {model_meta.get('best_score', 0):.4f}</p>
                <p style="color:#9AA0A6;">Task: {model_meta.get('task', 'N/A').title()}</p>
            </div>
            """, unsafe_allow_html=True)

        with champion_col2:
            st.markdown("**Top Features**")
            top_features = model_meta.get('top_features', [])
            if top_features:
                for i, feature in enumerate(top_features[:5], 1):
                    st.write(f"{i}. {feature}")
            else:
                st.write("No feature importance data available")

        # Model metrics (if available)
        if leaderboard and 'metrics' in leaderboard[0]:
            st.markdown("### 📊 **Detailed Metrics**")

            # Get best model metrics
            best_model_data = next((item for item in leaderboard if item['model'] == model_meta.get('best_model')), None)
            if best_model_data and 'metrics' in best_model_data:
                metrics = best_model_data['metrics']

                metrics_cols = st.columns(len(metrics))

                for i, (metric_name, value) in enumerate(metrics.items()):
                    with metrics_cols[i % len(metrics_cols)]:
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            st.metric(metric_name.replace('_', ' ').title(), f"{value:.4f}")
                        elif metric_name == 'confusion_matrix' and value:
                            st.write(f"**{metric_name.replace('_', ' ').title()}**")
                            cm_df = pd.DataFrame(value)
                            st.dataframe(cm_df, use_container_width=True)

    else:
        st.warning("🤖 No trained models yet")
        st.info("Go to the **Train** page to build your ML models")

# =====================================================
# TAB 4: MONITORING
# =====================================================
with tab_monitoring:
    st.markdown("## 🔍 **System Monitoring**")

    # System metrics
    st.markdown("### 💻 **System Resources**")

    monitor_col1, monitor_col2, monitor_col3, monitor_col4 = st.columns(4)

    with monitor_col1:
        st.metric("CPU Usage", "45%", "↓2%")

    with monitor_col2:
        st.metric("Memory Usage", "67%", "↑5%")

    with monitor_col3:
        st.metric("Disk Usage", "23%", "↓1%")

    with monitor_col4:
        st.metric("Network I/O", "1.2 MB/s", "↑0.1 MB/s")

    # Activity log
    st.markdown("### 📝 **Recent Activity Log**")

    activities = [
        {"time": "2024-01-09 14:32:15", "action": "Model training completed", "user": "Data Scientist", "status": "success"},
        {"time": "2024-01-09 14:28:42", "action": "EDA analysis finished", "user": "ML Engineer", "status": "success"},
        {"time": "2024-01-09 14:25:18", "action": "Dataset uploaded", "user": "Analyst", "status": "success"},
        {"time": "2024-01-09 14:20:05", "action": "ETL pipeline executed", "user": "Data Scientist", "status": "success"},
        {"time": "2024-01-09 14:15:33", "action": "Prediction batch completed", "user": "ML Engineer", "status": "warning"}
    ]

    activity_df = pd.DataFrame(activities)

    # Style the activity log
    def style_activity(row):
        if row['status'] == 'success':
            return ['background-color: rgba(16, 185, 129, 0.15); color: #a7f3d0'] * len(row)
        elif row['status'] == 'warning':
            return ['background-color: rgba(245, 158, 11, 0.15); color: #fde68a'] * len(row)
        else:
            return ['background-color: rgba(239, 68, 68, 0.15); color: #fca5a5'] * len(row)

    st.dataframe(
        activity_df.style.apply(style_activity, axis=1),
        use_container_width=True,
        hide_index=True
    )

    # Performance trends
    st.markdown("### 📈 **Performance Trends**")

    # Mock performance data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance_data = pd.DataFrame({
        'date': dates,
        'accuracy': [0.85 + 0.05 * (i % 3 - 1) + 0.01 * (i % 10) for i in range(30)],
        'latency': [120 + 10 * (i % 5 - 2) for i in range(30)],
        'requests': [100 + 20 * (i % 7 - 3) for i in range(30)]
    })

    # Performance chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['accuracy'],
        mode='lines+markers',
        name='Model Accuracy',
        line=dict(color='#0052cc', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=performance_data['date'],
        y=performance_data['latency'],
        mode='lines+markers',
        name='Response Time (ms)',
        line=dict(color='#2563eb', width=2),
        yaxis='y2'
    ))

    fig.update_layout(
        title='System Performance Over Time',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Accuracy', side='left'),
        yaxis2=dict(title='Latency (ms)', overlaying='y', side='right'),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
# =====================================================
# FOOTER
# =====================================================
page_footer()
