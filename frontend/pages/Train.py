# import streamlit as st
# import requests
# import pandas as pd

# st.set_page_config(page_title="AutoML Training", layout="wide")

# st.markdown("""
# <div style='background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;'>
#     <h1 style='font-size: 3rem; margin:0;'>🤖 <b>AutoML Training</b></h1>
#     <p style='font-size: 1.2rem; margin:5px;'>Multi-Model Competition • Best Model Auto-Selected</p>
# </div>
# """, unsafe_allow_html=True)

# if "dataset_id" not in st.session_state:
#     st.error("🚫 **Upload dataset first!**")
#     st.stop()

# dataset_id = st.session_state.dataset_id

# # ------------------ FETCH COLUMNS FOR TARGET SELECTION ------------------
# st.markdown("### 🎯 **Select Target Column**")
# try:
#     resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}")
#     if resp.status_code == 200:
#         eda_data = resp.json()["eda"]
#         all_columns = list(eda_data["missing"].keys())
#         # Prefer numeric/low-missing columns
#         numeric_columns = [
#             col for col in all_columns 
#             if eda_data["missing"].get(col, 999) < len(all_columns) * 0.3  # <30% missing
#         ]
#         target_options = numeric_columns or all_columns
#     else:
#         target_options = []
# except:
#     target_options = []

# if target_options:
#     selected_target = st.selectbox(
#         "Choose the column you want to predict:",
#         options=target_options,
#         index=0,
#         help="Best results with numeric targets and low missing values"
#     )
# else:
#     selected_target = st.text_input("Enter target column name manually:")

# # ------------------ TRAIN BUTTON ------------------
# st.markdown("### 🚀 **Launch Training**")
# col1, col2 = st.columns([1, 2])
# with col1:
#     st.info(f"**Target:** `{selected_target}`")
#     st.caption("Multi-model competition: RF vs XGBoost vs LightGBM")
# with col2:
#     train_button = st.button(
#         f"🚀 Train Models → Predict **{selected_target}**",
#         type="primary",
#         use_container_width=True
#     )

# if train_button:
#     with st.spinner("Running 3 models in parallel..."):
#         resp = requests.post(
#             f"http://127.0.0.1:8000/train/{dataset_id}",
#             json={"target": selected_target}
#         )
#         result = resp.json()

#     if result.get("status") == "ok":
#         st.session_state.model_data = result
#         st.session_state.model_trained = True
#         st.session_state.selected_target = selected_target

#         # ------------------ LEADERBOARD ------------------
#         st.markdown("## 🏆 **Model Leaderboard**")
#         leaderboard = result.get("model_leaderboard", {})
#         if leaderboard:
#             df_lb = pd.DataFrame([
#                 {"Model": k.upper(), "Score": v["score"]}
#                 for k, v in leaderboard.items()
#             ])
#             df_lb["Score"] = df_lb["Score"].round(4)
#             df_lb = df_lb.sort_values("Score", ascending=False).reset_index(drop=True)
#             df_lb.index += 1

#             st.dataframe(
#                 df_lb.style
#                 .format({"Score": "{:.4f}"})
#                 .background_gradient(subset=["Score"], cmap="Blues")
#                 .bar(subset=["Score"], color="#5fba7d"),
#                 use_container_width=True
#             )

#             winner = result.get("best_model", "Unknown").upper()
#             best_score = result.get("best_score", 0)
#             st.success(f"🏆 **{winner}** WINS with score **{best_score:.4f}**!")
#         else:
#             st.warning("No leaderboard data returned")

#         st.balloons()
#         st.rerun()
#     else:
#         st.error(f"❌ **Training failed**: {result.get('message', 'Unknown error')}")

# # ------------------ SHOW RESULTS IF TRAINED ------------------
# if st.session_state.get("model_trained"):
#     model_data = st.session_state.model_data

#     st.markdown("### 📊 **Training Summary**")
#     col1, col2, col3, col4 = st.columns(4)
#     with col1:
#         st.metric("🎯 Target", model_data.get("target", "N/A"))
#     with col2:
#         st.metric("🏆 Best Model", model_data.get("best_model", "N/A"))
#     with col3:
#         st.metric("📈 Best Score", f"{model_data.get('best_score', 0):.4f}")
#     with col4:
#         st.metric("🔧 Features Used", model_data.get("features_used", "N/A"))

#     st.success("✅ **Model ready!** Go to **Predict** page for live inference.")
#     st.caption("The best-performing model (RF/XGB/LGB) was automatically selected and saved.")

## pages/Train.py - COMPLETE FIXED VERSION (All imports included)
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

st.set_page_config(page_title="OmniSearch AI - AutoML Arena", layout="wide")

if "dataset_id" not in st.session_state:
    st.error("🚫 No dataset loaded. Go to **Upload** page first.")
    st.stop()

dataset_id = st.session_state.dataset_id

# HEADER
st.markdown("""
<div style='background: linear-gradient(90deg, #8b0000 0%, #ff4500 50%, #1e3a72 100%); 
           padding: 2.5rem; border-radius: 20px; color: white; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.4);'>
    <h1 style='font-size: 3.5rem; margin: 0;'>⚔️ <b>AutoML Arena</b></h1>
    <p style='font-size: 1.4rem; opacity: 0.95;'>Random Forest vs XGBoost vs LightGBM • Live Competition</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def get_model_status():
    try:
        resp = requests.get(f"http://127.0.0.1:8000/meta/{dataset_id}", timeout=10)
        return resp.json() if resp.status_code == 200 else None
    except Exception as e:
        st.error(f"Backend connection failed: {str(e)}")
        return None

model_status = get_model_status()

# MAIN BUTTON
col1, col2 = st.columns([3, 1])

with col1:
    if model_status and model_status.get('status') == 'ok':
        best_score = max([m.get('primary_score', 0) for m in model_status.get('model_leaderboard', {}).values()])
        best_model = model_status.get('best_model', 'N/A')
        st.success(f"✅ **BATTLE COMPLETE!** 🏆 {best_model} wins with {best_score:.1%}")
    else:
        if st.button("🚀 **START MODEL BATTLE**", type="primary", use_container_width=True, help="Trains 3 models & picks the best"):
            with st.spinner("⚔️ Running model competition (RF vs XGB vs LGB)..."):
                try:
                    resp = requests.post(f"http://127.0.0.1:8000/train/{dataset_id}", timeout=180)
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get('status') == 'ok':
                            st.success(f"🏆 **{result['best_model']} WINS** with {result['best_score']:.1%}!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(f"Training failed: {result.get('message', 'Unknown error')}")
                    else:
                        st.error(f"Backend error {resp.status_code}: {resp.text[:200]}")
                except requests.exceptions.Timeout:
                    st.error("⏰ Training timeout (180s). Try smaller dataset or check backend.")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

with col2:
    if model_status:
        st.metric("📊 Features", len(model_status.get('features', [])))
        st.metric("🎯 Target", model_status.get('target', 'Auto'))
        st.metric("📈 Task", model_status.get('task', '?').upper())

if not model_status or model_status.get('status') != 'ok':
    st.info("""
    **👆 Click "START MODEL BATTLE" to:**
    • Train **3 models** (RF, XGBoost, LightGBM)  
    • **20% test set** validation
    • **Auto-select BEST model**
    • **Save for predictions**
    """)
    st.stop()

# ========== RESULTS DASHBOARD ==========
results = model_status['model_leaderboard']
best_model_name = model_status['best_model'].lower()

st.markdown("---")
st.markdown("### 🏆 **LIVE LEADERBOARD**")
leaderboard_data = []
for model_name, data in results.items():
    leaderboard_data.append({
        '🥇🥈🥉': '🥇' if model_name == best_model_name else '🥈' if model_name == list(results.keys())[1] else '🥉',
        'Model': model_name.upper(),
        'Score': f"{data['primary_score']:.1%}",
        'Test Set': f"{data['test_samples']:,}",
        'Train Set': f"{data['train_samples']:,}"
    })

lb_df = pd.DataFrame(leaderboard_data)
st.dataframe(lb_df, use_container_width=True)

# ========== VISUAL COMPARISON ==========
st.markdown("### 📊 **BATTLE ARENA**")
models = list(results.keys())
scores = [results[m]['primary_score'] for m in models]
colors = ['#FFD700', '#C0C0C0', '#CD7F32']  # Gold, Silver, Bronze

fig = go.Figure()
for i, model in enumerate(models):
    fig.add_trace(go.Bar(
        name=model.upper(),
        x=[model],
        y=[scores[i]],
        marker_color=colors[i],
        text=[f"{scores[i]:.1%}"],
        textposition="auto"
    ))

fig.update_layout(
    title="🏆 Primary Score Competition (20% Test Set)",
    yaxis_title="Score",
    height=500,
    showlegend=True,
    bargap=0.3
)
st.plotly_chart(fig, use_container_width=True)

# ========== MODEL CARDS ==========
st.markdown("### 🏅 **MODEL BREAKDOWN**")
cols = st.columns(len(models))
for i, model_name in enumerate(models):
    with cols[i]:
        data = results[model_name]
        is_best = model_name == best_model_name
        
        st.markdown(f"### **{model_name.upper()}** {'🏆' if is_best else ''}")
        
        # Primary metric
        primary = data['primary_score']
        st.metric("Score", f"{primary:.3f}", delta="🏆 BEST!" if is_best else None)
        
        # Key metrics table
        metrics = {k: v for k, v in data['all_metrics'].items() if k != 'confusion_matrix'}
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value']).round(3)
        st.dataframe(metrics_df, use_container_width=True, height=180)
        
        st.caption(f"Test: {data['test_samples']:,} samples")

# ========== BEST MODEL DEEP DIVE ==========
st.markdown("### 🎯 **CHAMPION ANALYSIS**")
best_data = results[best_model_name]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🥇 CHAMPION", best_model_name.upper())
with col2:
    st.metric("Score", f"{best_data['primary_score']:.3f}")
with col3:
    st.metric("Test Size", f"{best_data['test_samples']:,}")

# Truth vs Prediction
st.markdown("### ✅ **TRUTH vs PREDICTION** (First 10)")
tvp_df = pd.DataFrame(best_data['true_vs_pred'], columns=['True', 'Predicted']).round(2)
st.dataframe(tvp_df, use_container_width=True)

# Confusion Matrix (if classification)
if 'confusion_matrix' in best_data['all_metrics']:
    st.markdown("### 📈 **CONFUSION MATRIX**")
    cm = pd.DataFrame(best_data['all_metrics']['confusion_matrix'])
    fig_cm = px.imshow(
        cm, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale='Blues',
        title=f"{best_model_name.upper()} Confusion Matrix"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ========== NEXT STEPS ==========
st.markdown("### 🚀 **PRODUCTION READY**")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔮 **LIVE PREDICTIONS**", type="primary", use_container_width=True):
        st.switch_page("pages/Predict.py")
with col2:
    st.download_button(
        "💾 **Export Results**", 
        data=json.dumps(model_status, indent=2, default=str),
        file_name=f"AutoML_Results_{dataset_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

st.markdown("---")
st.caption(f"⚡ Trained on {model_status.get('trained_on', 'N/A')} | {len(model_status.get('features', []))} features | {model_status.get('task', '?').upper()}")
