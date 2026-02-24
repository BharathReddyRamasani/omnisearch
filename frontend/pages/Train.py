import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
from datetime import datetime

API = "http://127.0.0.1:8001/api"

st.set_page_config(
    page_title="OmniSearch AI – Industrial AutoML Training",
    layout="wide",
    initial_sidebar_state="expanded"
)

from theme import inject_theme, page_header, page_footer
inject_theme()

# =====================================================
# DATASET CHECK
# =====================================================
dataset_id = st.session_state.get("dataset_id")
columns = st.session_state.get("columns")

if not dataset_id or not columns:
    st.error("🚫 No dataset loaded. Upload a dataset first.")
    st.stop()

# =====================================================
# SIDEBAR CONFIG
# =====================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")
    
    # Model selection
    st.markdown("**Algorithms to Train:**")
    train_regression = st.checkbox("Regression Models", value=True, 
                                   help="Linear, Polynomial, Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting")
    train_classification = st.checkbox("Classification Models", value=True,
                                       help="Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes, SVM, KNN")
    
    st.markdown("---")
    st.markdown("**Training Parameters:**")
    test_size = st.slider("Test Size (%)", 10, 50, 20, 5)
    random_state = st.number_input("Random State", value=42, min_value=0)
    
    st.markdown("---")
    st.caption("Industrial ML Pipeline • v2.0")

# =====================================================
# HEADER
# =====================================================
page_header("🚀", "Industrial AutoML Training", "10+ Algorithms • Advanced Metrics • Enterprise-Grade")

st.markdown("---")

# =====================================================
# WORKFLOW STATUS
# =====================================================
st.markdown("### 🔄 ML Pipeline Workflow")
col_wf1, col_wf2, col_wf3, col_wf4 = st.columns(4)

with col_wf1:
    st.markdown("✅ **Upload**\nData Ingestion")
with col_wf2:
    st.markdown("📊 **EDA**\nExplore Data")
with col_wf3:
    st.markdown("🧹 **ETL**\nClean Data")
with col_wf4:
    st.markdown("⚡ **Train**\nBuild Models")

st.markdown("---")

# =====================================================
# TABS
# =====================================================
tab_setup, tab_training, tab_results, tab_advanced = st.tabs([
    "🎯 Setup", "⚡ Training", "📊 Results", "🔬 Advanced"
])

# =====================================================
# TAB 1: SETUP
# =====================================================
with tab_setup:
    st.markdown("## 🎯 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Target Variable")
        
        # ✅ Filter out ID-like columns (>99% unique)
        filterable_columns = []
        if columns:
            try:
                sample_resp = requests.get(f"{API}/datasets/{dataset_id}/sample", timeout=10)
                if sample_resp.status_code == 200:
                    sample_data = sample_resp.json()
                    sample_df = pd.DataFrame(sample_data)
                    
                    for col in columns:
                        if col in sample_df.columns:
                            unique_pct = sample_df[col].nunique() / len(sample_df) * 100
                            # Exclude ID columns (>95% unique)
                            if unique_pct < 95:
                                filterable_columns.append(col)
                        else:
                            filterable_columns.append(col)
            except:
                filterable_columns = columns
        
        # Use filtered columns if available, otherwise all columns
        target_options = filterable_columns if filterable_columns else columns
        
        target = st.selectbox(
            "Select target column",
            options=target_options,
            help="The column to predict. ID/name columns (>95% unique) are excluded. Must be numeric for regression, categorical for classification."
        )
        
        # Auto-detect task
        if target:
            # Sample data to detect task
            try:
                sample_resp = requests.get(f"{API}/datasets/{dataset_id}/sample", timeout=10)
                if sample_resp.status_code == 200:
                    sample_data = sample_resp.json()
                    sample_df = pd.DataFrame(sample_data)
                    if target in sample_df.columns:
                        unique_vals = sample_df[target].nunique()
                        detected_task = "Classification" if unique_vals <= 15 else "Regression"
                        st.info(f"🔍 Detected Task: **{detected_task}** ({unique_vals} unique values)")
            except:
                pass
    
    with col2:
        st.markdown("### Dataset Overview")
        
        # Get dataset info
        try:
            info_resp = requests.get(f"{API}/datasets/{dataset_id}/info", timeout=10)
            if info_resp.status_code == 200:
                info = info_resp.json()
                st.metric("Total Rows", info.get("rows", "N/A"))
                st.metric("Total Columns", info.get("columns", "N/A"))
                st.metric("Data Types", f"{info.get('dtypes', {})}")
        except:
            st.warning("Could not fetch dataset info")
    
    st.markdown("---")
    st.markdown("### Algorithm Selection")
    
    # Show selected algorithms
    selected_models = []
    if train_regression:
        selected_models.extend([
            "LinearRegression", "PolynomialRegression", "Ridge", "Lasso", 
            "DecisionTree", "RandomForest", "GradientBoosting"
        ])
    if train_classification:
        selected_models.extend([
            "LogisticRegression", "DecisionTree", "RandomForest", "GradientBoosting",
            "NaiveBayes", "SVM", "KNN"
        ])
    
    st.write(f"**{len(selected_models)} models selected:** {', '.join(selected_models[:5])}{'...' if len(selected_models) > 5 else ''}")

# =====================================================
# TAB 2: TRAINING
# =====================================================
with tab_training:
    st.markdown("## ⚡ Model Training")
    
    # ✅ CHECK: Verify clean data exists before training
    clean_data_available = False
    try:
        check_resp = requests.get(f"{API}/datasets/{dataset_id}/download/clean", timeout=10)
        clean_data_available = check_resp.status_code == 200
    except:
        clean_data_available = False
    
    if not clean_data_available:
        st.warning(
            """
            ⚠️ **Clean data not available** - Training requires cleaned data to prevent data leakage.
            
            **Please follow the workflow:**
            1. 📊 **EDA** - Analyze your raw data
            2. 🧹 **ETL** - Clean and prepare your data
            3. ⚡ **Train** - Train models on clean data
            
            After completing ETL, come back here to train your models.
            """
        )
        st.info("💡 Go to the **ETL** page to clean your data first.")
    else:
        st.success("✅ Clean data available - Ready to train!")
    
    st.markdown("---")
    
    # Training button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Start Industrial Training", type="primary", use_container_width=True, disabled=not clean_data_available):
            if not clean_data_available:
                st.error("❌ Cannot train: Clean data not available. Please run ETL first.")
                st.stop()
            
            # ✅ PRE-FLIGHT VALIDATION
            # Check that target is valid and not an ID column
            if not target:
                st.error("❌ Target variable not selected. Please choose a target column.")
                st.stop()
            
            if target not in columns:
                st.error(f"❌ Target '{target}' not found in dataset columns.")
                st.stop()
            
            # Validate target is not an ID column (>95% unique)
            try:
                sample_resp = requests.get(f"{API}/datasets/{dataset_id}/sample", timeout=10)
                if sample_resp.status_code == 200:
                    sample_data = sample_resp.json()
                    sample_df = pd.DataFrame(sample_data)
                    if target in sample_df.columns:
                        unique_pct = sample_df[target].nunique() / len(sample_df) * 100
                        if unique_pct > 95:
                            st.error(f"❌ Invalid target: '{target}' has {unique_pct:.1f}% unique values (looks like an ID column). Please select a different column.")
                            st.stop()
            except:
                pass
            
            # ✅ Determine task based on target variable
            # If both regression and classification are selected, auto-detect
            # If only one is selected, use that
            task = None  # Auto-detect
            
            if train_regression and not train_classification:
                task = "regression"
            elif train_classification and not train_regression:
                task = "classification"
            # else: None (auto-detect based on target dtype)
            
            payload = {
                "target": target,
                "test_size": test_size / 100,
                "random_state": random_state,
                "task": task  # ✅ Use correct parameter name
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🔬 Training industrial ML models..."):
                try:
                    status_text.text("Submitting training job...")
                    progress_bar.progress(5)
                    
                    start_time = time.time()
                    
                    # Submit training job (async)
                    resp = requests.post(
                        f"{API}/train/{dataset_id}",
                        json=payload,
                        timeout=30
                    )
                    
                    if resp.status_code != 200:
                        st.error(f"Backend error: {resp.status_code}")
                        st.json(resp.json() if resp.content else resp.text)
                        st.stop()
                    
                    job_response = resp.json()
                    
                    if job_response.get("status") != "accepted":
                        st.error("Failed to submit training job")
                        st.json(job_response)
                        st.stop()
                    
                    job_id = job_response.get("job_id")
                    st.info(f"📋 Training job submitted: {job_id}")
                    st.info("Polling job status... (this may take a few minutes)")
                    
                    progress_bar.progress(10)
                    
                    # Poll job status
                    max_polls = 120  # 10 minutes with 5-second intervals
                    poll_count = 0
                    
                    while poll_count < max_polls:
                        try:
                            job_resp = requests.get(f"{API}/jobs/{job_id}", timeout=10)
                            
                            if job_resp.status_code != 200:
                                st.warning(f"Could not fetch job status: {job_resp.status_code}")
                                time.sleep(5)
                                poll_count += 1
                                continue
                            
                            job_data = job_resp.json()
                            job_status = job_data.get("status")
                            job_progress = job_data.get("progress", poll_count // 6)
                            job_message = job_data.get("message", "Training...")
                            
                            status_text.text(f"Status: {job_message}")
                            progress_bar.progress(min(99, 10 + (job_progress // 2)))
                            
                            if job_status == "completed":
                                result = job_data.get("result")
                                
                                if result and result.get("status") == "ok":
                                    # Store results
                                    st.session_state.model_meta = result
                                    st.session_state.training_time = time.time() - start_time
                                    
                                    progress_bar.progress(100)
                                    status_text.text("✅ Training completed successfully!")
                                    
                                    st.success(f"🏆 Training completed in {st.session_state.training_time:.1f}s")
                                    st.balloons()
                                    
                                    # Auto-switch to results tab
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Training completed but returned error status")
                                    st.json(result)
                                    st.stop()
                                break
                            
                            elif job_status == "failed":
                                st.error(f"❌ Training failed: {job_data.get('message', 'Unknown error')}")
                                
                                error_details = job_data.get("error") or job_data.get("details", "")
                                error_code = job_data.get("error_code")
                                recommendation = job_data.get("recommendation")
                                workflow_steps = job_data.get("workflow_steps", [])
                                
                                if error_code == "CLEAN_DATA_REQUIRED":
                                    st.warning(
                                        """
                                        ⚠️ **Clean Data Required**
                                        
                                        Training requires cleaned data to prevent data leakage.
                                        """
                                    )
                                    if workflow_steps:
                                        st.markdown("**Required Workflow Steps:**")
                                        for step in workflow_steps:
                                            st.markdown(f"- {step}")
                                    if recommendation:
                                        st.info(f"💡 {recommendation}")
                                    
                                    st.markdown("**Next Step:** Go to **ETL** page to clean your data, then return here to train.")
                                elif error_details:
                                    st.error(f"Error details: {error_details}")
                                
                                if recommendation and error_code != "CLEAN_DATA_REQUIRED":
                                    st.info(f"💡 {recommendation}")
                                
                                st.stop()
                                break
                            
                            # Job still running
                            time.sleep(5)
                            poll_count += 1
                        
                        except requests.exceptions.Timeout:
                            st.warning("Job status request timed out, retrying...")
                            time.sleep(2)
                            poll_count += 2
                            continue
                    
                    if poll_count >= max_polls:
                        st.error("⏰ Training timed out (10min). The job may still be running in background.")
                        st.info(f"Check status later with job ID: {job_id}")
                        st.stop()
                    
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timed out. The job may be running in background.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                finally:
                    progress_bar.empty()
                    status_text.empty()
    
    with col2:
        st.markdown("### Training Info")
        st.metric("Models", len(selected_models))
        st.metric("Test Size", f"{test_size}%")
        st.metric("Random State", random_state)
        
        if "training_time" in st.session_state:
            st.metric("Last Training", f"{st.session_state.training_time:.1f}s")

# =====================================================
# LOAD TRAINING RESULTS
# =====================================================
model_meta = st.session_state.get("model_meta")

if not model_meta:
    with tab_results:
        st.info("👆 Configure and start training to see results")
    with tab_advanced:
        st.info("👆 Train models first to access advanced analytics")
else:
    # =====================================================
    # TAB 3: RESULTS
    # =====================================================
    with tab_results:
        st.markdown("## 📊 Training Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🏆 Best Model", model_meta.get("best_model", "N/A"))
        
        with col2:
            # ✅ Safe access with fallback chain
            best_score = model_meta.get('best_score')
            if best_score is None:
                # Fallback to ranking_score from leaderboard
                leaderboard = model_meta.get('leaderboard', [])
                if leaderboard:
                    best_score = max([m.get('ranking_score', m.get('holdout_score', 0)) for m in leaderboard], default=0)
                else:
                    best_score = 0
            st.metric("📈 Best Score", f"{float(best_score):.4f}")
        
        with col3:
            st.metric("🎯 Task", model_meta.get("task", "N/A").upper())
        
        with col4:
            st.metric("⏱️ Trained", model_meta.get("trained_at", "N/A"))
        
        st.markdown("---")
        
        # Leaderboard
        st.markdown("### 🏆 Model Leaderboard")
        
        leaderboard = model_meta.get("leaderboard", [])
        if leaderboard:
            df_lb = pd.DataFrame(leaderboard)
            
            # Handle columns: ranking_score is used for ranking, holdout_score is test performance
            # Determine which score column to display
            score_col = None
            if "ranking_score" in df_lb.columns:
                score_col = "ranking_score"
            elif "holdout_score" in df_lb.columns:
                score_col = "holdout_score"
            elif "score" in df_lb.columns:
                score_col = "score"
            
            # Style the dataframe - highlight best model
            def highlight_best(s):
                if score_col and score_col in df_lb.columns:
                    is_best = s.name == df_lb[score_col].idxmax()
                    return ['background-color: #e6f7ff' if is_best else '' for _ in s]
                return ['' for _ in s]
            
            # Prepare display columns and formatting
            display_cols = ["model"]
            format_dict = {}
            
            if "ranking_score" in df_lb.columns:
                display_cols.append("ranking_score")
                format_dict["ranking_score"] = "{:.4f}"
            if "holdout_score" in df_lb.columns:
                display_cols.append("holdout_score")
                format_dict["holdout_score"] = "{:.4f}"
            
            if "train_rows" in df_lb.columns:
                display_cols.append("train_rows")
                format_dict["train_rows"] = "{:,}"
            if "test_rows" in df_lb.columns:
                display_cols.append("test_rows")
                format_dict["test_rows"] = "{:,}"
            
            if "training_time_seconds" in df_lb.columns:
                display_cols.append("training_time_seconds")
                format_dict["training_time_seconds"] = "{:.2f}"
            
            # Select columns to display
            df_display = df_lb[display_cols]
            
            # Apply styling
            styled_df = df_display.style.apply(highlight_best, axis=0)
            if format_dict:
                styled_df = styled_df.format(format_dict)
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Performance comparison chart
            st.markdown("### 📊 Performance Comparison")
            
            # Determine which score column to use for chart
            chart_score_col = score_col if score_col else "ranking_score"
            if chart_score_col not in df_lb.columns:
                # Fallback if score_col was determined but somehow not in columns
                chart_score_col = "holdout_score" if "holdout_score" in df_lb.columns else "score"
            
            if chart_score_col in df_lb.columns:
                fig = px.bar(
                    df_lb,
                    x="model",
                    y=chart_score_col,
                    color="model",
                    text=chart_score_col,
                    title=f"Model Performance ({chart_score_col.replace('_', ' ').title()})",
                    labels={"model": "Algorithm", chart_score_col: "Score"}
                )
                fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig.update_layout(showlegend=False, height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No score column available for visualization")
        
        # Model details
        st.markdown("### 🔍 Model Details")
        
        model_details = model_meta.get("model_details", [])
        if model_details:
            selected_model = st.selectbox(
                "Select model for detailed analysis",
                options=[m["model"] for m in model_details],
                index=0
            )
            
            for detail in model_details:
                if detail["model"] == selected_model:
                    metrics = detail.get("metrics", {})
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**{selected_model} Metrics**")
                        for key, value in metrics.items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                st.metric(key.replace("_", " ").title(), f"{value:.4f}")
                            elif key == "confusion_matrix" and value:
                                st.write("Confusion Matrix:")
                                cm_df = pd.DataFrame(value)
                                st.dataframe(cm_df)
                    
                    with col2:
                        st.markdown("**Sample Predictions**")
                        preds = detail.get("predictions_sample", [])
                        if preds:
                            st.write(f"First 10 predictions: {preds}")
                    
                    break
    
    # =====================================================
    # TAB 4: ADVANCED
    # =====================================================
    with tab_advanced:
        st.markdown("## 🔬 Advanced Analytics")
        
        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        
        top_features = model_meta.get("top_features", [])
        if top_features:
            st.write("Top features for best model:")
            for i, feat in enumerate(top_features[:10], 1):
                st.write(f"{i}. {feat}")
        
        # Dropped columns
        dropped_cols = model_meta.get("dropped_id_columns", [])
        if dropped_cols:
            st.markdown("### 🗑️ Dropped ID Columns")
            st.write("Automatically removed unique identifier columns:")
            for col in dropped_cols:
                st.write(f"- {col}")
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                "📄 Download Full Report (JSON)",
                data=json.dumps(model_meta, indent=2),
                file_name=f"industrial_automl_report_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Generate summary report
            summary = {
                "dataset_id": dataset_id,
                "target": model_meta.get("target"),
                "task": model_meta.get("task"),
                "best_model": model_meta.get("best_model"),
                "best_score": model_meta.get("best_score"),
                "models_trained": len(model_meta.get("leaderboard", [])),
                "training_time": st.session_state.get("training_time", "N/A"),
                "trained_at": model_meta.get("trained_at")
            }
            
            st.download_button(
                "📊 Download Summary (JSON)",
                data=json.dumps(summary, indent=2),
                file_name=f"summary_report_{dataset_id}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Raw metadata
        with st.expander("🔧 Raw Training Metadata"):
            st.json(model_meta)

# =====================================================
# FOOTER
# =====================================================
page_footer()