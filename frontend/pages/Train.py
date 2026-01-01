import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="OmniSearch AI - AutoML Training",
    layout="wide"
)

# ---------------- DATASET CHECK ----------------
if "dataset_id" not in st.session_state or not st.session_state.dataset_id:
    st.error("🚫 No dataset loaded. Go to **Upload** page first.")
    st.stop()

dataset_id = st.session_state.dataset_id

# ---------------- HEADER ----------------
st.markdown(
    """
<div style='background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%);
            padding: 2.5rem; border-radius: 18px; color: white;
            text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <h1 style='font-size: 3.2rem; margin:0;'>🤖 <b>AutoML Training Arena</b></h1>
    <p style='font-size: 1.3rem; margin-top:8px; opacity:0.95;'>
        RF vs XGBoost vs LightGBM • Best Model Auto-Selected
    </p>
</div>
""",
    unsafe_allow_html=True
)

# ---------------- FETCH MODEL STATUS ----------------
@st.cache_data(ttl=30)
def get_model_meta():
    try:
        r = requests.get(f"{API}/meta/{dataset_id}", timeout=10)
        return r.json() if r.status_code == 200 else None
    except:
        return None

model_meta = get_model_meta()

# ---------------- TARGET SELECTION ----------------
st.markdown("### 🎯 Target Column")

target_col = st.text_input(
    "Enter target column (leave empty for automatic selection)",
    help="If empty, backend selects the best numeric target automatically"
)

# ---------------- TRAIN BUTTON ----------------
col1, col2 = st.columns([3, 1])

with col1:
    if st.button(
        "🚀 Start Model Battle",
        type="primary",
        use_container_width=True
    ):
        payload = {"target": target_col} if target_col else {}

        with st.spinner("⚔️ Training models (RF vs XGB vs LGB)..."):
            try:
                resp = requests.post(
                    f"{API}/train/{dataset_id}",
                    json=payload,
                    timeout=180
                )

                if resp.status_code != 200:
                    st.error(f"Backend error: {resp.status_code}")
                    st.stop()

                result = resp.json()
                if result.get("status") != "ok":
                    st.error(result.get("message", "Training failed"))
                    st.stop()

                st.session_state.last_training = result
                st.success(
                    f"🏆 {result['best_model']} WINS with score {result['best_score']:.4f}"
                )
                st.balloons()
                st.cache_data.clear()
                st.rerun()

            except requests.exceptions.Timeout:
                st.error("⏰ Training timed out (180s). Try a smaller dataset.")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

with col2:
    if model_meta and model_meta.get("status") == "ok":
        st.metric("🎯 Target", model_meta.get("target", "Auto"))
        st.metric("📈 Task", model_meta.get("task", "").upper())
        st.metric("🔧 Features", len(model_meta.get("features", [])))

# ---------------- NO MODEL YET ----------------
if not model_meta or model_meta.get("status") != "ok":
    st.info(
        """
👆 **Click “Start Model Battle” to:**
- Train **3 models** (RF, XGBoost, LightGBM)
- Use **20% test split**
- Automatically select **best model**
- Save it for predictions
"""
    )
    st.stop()
# ---------------- RESULTS DASHBOARD ----------------
results = model_meta.get("leaderboard", {})

if not results:
    st.error("No leaderboard data found. Train the model first.")
    st.stop()

best_model = model_meta.get("best_model", "").lower()
best_score = model_meta.get("best_score")

st.markdown("---")
st.markdown("## 🏆 Model Leaderboard")

lb_df = (
    pd.DataFrame(
        [{"Model": k.upper(), "Score": v} for k, v in results.items()]
    )
    .sort_values("Score", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    lb_df.style
    .background_gradient(subset=["Score"], cmap="Blues")
    .format({"Score": "{:.4f}"}),
    use_container_width=True
)

# ---------------- SCORE COMPARISON ----------------
st.markdown("## 📊 Score Comparison")

fig = px.bar(
    lb_df,
    x="Model",
    y="Score",
    color="Model",
    text="Score",
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

# ---------------- BEST MODEL DETAILS ----------------
st.markdown("## 🥇 Champion Model")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Champion", best_model.upper())

with c2:
    st.metric("Best Score", f"{best_score:.4f}")

with c3:
    st.metric("Task", model_meta.get("task", "").upper())

st.info(
    f"""
**Why this model won**
- Highest validation score on unseen data
- Robust to feature scaling & missing values
- Suitable for enterprise production inference
"""
)

# ---------------- EXPORT + NEXT ----------------
st.markdown("---")
st.markdown("## 🚀 Next Steps")

n1, n2 = st.columns(2)

with n1:
    if st.button("🔮 Go to Predict", type="primary", use_container_width=True):
        st.switch_page("pages/5_Predict.py")

with n2:
    st.download_button(
        "💾 Export Training Report",
        data=json.dumps(model_meta, indent=2),
        file_name=f"AutoML_Report_{dataset_id}.json",
        mime="application/json",
        use_container_width=True,
    )

st.caption(
    f"⚡ Trained at {model_meta.get('trained_at', 'N/A')} | "
    f"{len(model_meta.get('features', [])) if model_meta.get('features') else 'N/A'} features | "
    f"{model_meta.get('task', '').upper()}"
)
