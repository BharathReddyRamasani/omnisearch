# # import streamlit as st
# # import requests
# # import pandas as pd
# # import plotly.express as px
# # import json

# # API = "http://127.0.0.1:8000/api"

# # st.set_page_config(
# #     page_title="OmniSearch AI - AutoML Training",
# #     layout="wide"
# # )

# # # ---------------- DATASET CHECK ----------------
# # if "dataset_id" not in st.session_state or not st.session_state.dataset_id:
# #     st.error("🚫 No dataset loaded. Go to **Upload** page first.")
# #     st.stop()

# # dataset_id = st.session_state.dataset_id

# # # ---------------- HEADER ----------------
# # st.markdown(
# #     """
# # <div style='background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%);
# #             padding: 2.5rem; border-radius: 18px; color: white;
# #             text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
# #     <h1 style='font-size: 3.2rem; margin:0;'>🤖 <b>AutoML Training Arena</b></h1>
# #     <p style='font-size: 1.3rem; margin-top:8px; opacity:0.95;'>
# #         RF vs XGBoost vs LightGBM • Best Model Auto-Selected
# #     </p>
# # </div>
# # """,
# #     unsafe_allow_html=True
# # )

# # # ---------------- FETCH MODEL STATUS ----------------
# # @st.cache_data(ttl=30)
# # def get_model_meta():
# #     try:
# #         r = requests.get(f"{API}/meta/{dataset_id}", timeout=10)
# #         return r.json() if r.status_code == 200 else None
# #     except:
# #         return None

# # model_meta = get_model_meta()

# # # ---------------- TARGET SELECTION ----------------
# # st.markdown("### 🎯 Target Column")

# # target_col = st.text_input(
# #     "Enter target column (leave empty for automatic selection)",
# #     help="If empty, backend selects the best numeric target automatically"
# # )

# # # ---------------- TRAIN BUTTON ----------------
# # col1, col2 = st.columns([3, 1])

# # with col1:
# #     if st.button(
# #         "🚀 Start Model Battle",
# #         type="primary",
# #         use_container_width=True
# #     ):
# #         payload = {"target": target_col} if target_col else {}

# #         with st.spinner("⚔️ Training models (RF vs XGB vs LGB)..."):
# #             try:
# #                 resp = requests.post(
# #                     f"{API}/train/{dataset_id}",
# #                     json=payload,
# #                     timeout=180
# #                 )

# #                 if resp.status_code != 200:
# #                     st.error(f"Backend error: {resp.status_code}")
# #                     st.stop()

# #                 result = resp.json()
# #                 if result.get("status") != "ok":
# #                     st.error(result.get("message", "Training failed"))
# #                     st.stop()

# #                 st.session_state.last_training = result
# #                 st.success(
# #                     f"🏆 {result['best_model']} WINS with score {result['best_score']:.4f}"
# #                 )
# #                 st.balloons()
# #                 st.cache_data.clear()
# #                 st.rerun()

# #             except requests.exceptions.Timeout:
# #                 st.error("⏰ Training timed out (180s). Try a smaller dataset.")
# #             except Exception as e:
# #                 st.error(f"Connection error: {str(e)}")

# # with col2:
# #     if model_meta and model_meta.get("status") == "ok":
# #         st.metric("🎯 Target", model_meta.get("target", "Auto"))
# #         st.metric("📈 Task", model_meta.get("task", "").upper())
# #         st.metric("🔧 Features", len(model_meta.get("features", [])))

# # # ---------------- NO MODEL YET ----------------
# # if not model_meta or model_meta.get("status") != "ok":
# #     st.info(
# #         """
# # 👆 **Click “Start Model Battle” to:**
# # - Train **3 models** (RF, XGBoost, LightGBM)
# # - Use **20% test split**
# # - Automatically select **best model**
# # - Save it for predictions
# # """
# #     )
# #     st.stop()
# # # ---------------- RESULTS DASHBOARD ----------------
# # results = model_meta.get("leaderboard", {})

# # if not results:
# #     st.error("No leaderboard data found. Train the model first.")
# #     st.stop()

# # best_model = model_meta.get("best_model", "").lower()
# # best_score = model_meta.get("best_score")

# # st.markdown("---")
# # st.markdown("## 🏆 Model Leaderboard")

# # lb_df = (
# #     pd.DataFrame(
# #         [{"Model": k.upper(), "Score": v} for k, v in results.items()]
# #     )
# #     .sort_values("Score", ascending=False)
# #     .reset_index(drop=True)
# # )

# # st.dataframe(
# #     lb_df.style
# #     .background_gradient(subset=["Score"], cmap="Blues")
# #     .format({"Score": "{:.4f}"}),
# #     use_container_width=True
# # )

# # # ---------------- SCORE COMPARISON ----------------
# # st.markdown("## 📊 Score Comparison")

# # fig = px.bar(
# #     lb_df,
# #     x="Model",
# #     y="Score",
# #     color="Model",
# #     text="Score",
# # )
# # fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
# # fig.update_layout(height=420)
# # st.plotly_chart(fig, use_container_width=True)

# # # ---------------- BEST MODEL DETAILS ----------------
# # st.markdown("## 🥇 Champion Model")

# # c1, c2, c3 = st.columns(3)

# # with c1:
# #     st.metric("Champion", best_model.upper())

# # with c2:
# #     st.metric("Best Score", f"{best_score:.4f}")

# # with c3:
# #     st.metric("Task", model_meta.get("task", "").upper())

# # st.info(
# #     f"""
# # **Why this model won**
# # - Highest validation score on unseen data
# # - Robust to feature scaling & missing values
# # - Suitable for enterprise production inference
# # """
# # )

# # # ---------------- EXPORT + NEXT ----------------
# # st.markdown("---")
# # st.markdown("## 🚀 Next Steps")

# # n1, n2 = st.columns(2)

# # with n1:
# #     if st.button("🔮 Go to Predict", type="primary", use_container_width=True):
# #         st.switch_page("pages/5_Predict.py")

# # with n2:
# #     st.download_button(
# #         "💾 Export Training Report",
# #         data=json.dumps(model_meta, indent=2),
# #         file_name=f"AutoML_Report_{dataset_id}.json",
# #         mime="application/json",
# #         use_container_width=True,
# #     )

# # st.caption(
# #     f"⚡ Trained at {model_meta.get('trained_at', 'N/A')} | "
# #     f"{len(model_meta.get('features', [])) if model_meta.get('features') else 'N/A'} features | "
# #     f"{model_meta.get('task', '').upper()}"
# # )

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px

# API = "http://127.0.0.1:8000/api"

# st.set_page_config(page_title="Enterprise AutoML", layout="wide")

# dataset_id = st.session_state.get("dataset_id")
# if not dataset_id:
#     st.error("Upload dataset first")
#     st.stop()

# st.title("🤖 Enterprise AutoML Training")

# # ---------------- TARGET ----------------
# target = st.selectbox(
#     "Select target column",
#     options=st.session_state.get("columns", [])
# )

# # ---------------- TRAIN ----------------
# if st.button("🚀 Train Models", type="primary"):
#     with st.spinner("Training models..."):
#         resp = requests.post(
#             f"{API}/train/{dataset_id}",
#             json={"target": target},
#             timeout=300
#         )
#         if resp.status_code != 200:
#             st.error(resp.text)
#             st.stop()

#         st.session_state.model_meta = resp.json()
#         st.success("Training complete")

# meta = st.session_state.get("model_meta")
# if not meta:
#     st.stop()

# # ---------------- LEADERBOARD ----------------
# st.subheader("🏆 Model Leaderboard")

# rows = []
# for m, score in meta["leaderboard"].items():
#     rows.append({"Model": m.upper(), "Score": score})

# df = pd.DataFrame(rows).sort_values("Score", ascending=False)
# st.dataframe(df, use_container_width=True)

# fig = px.bar(df, x="Model", y="Score", color="Model")
# st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json

API = "http://127.0.0.1:8000/api"

st.set_page_config(
    page_title="OmniSearch AI – Enterprise AutoML",
    layout="wide"
)

# =====================================================
# DATASET CHECK
# =====================================================
dataset_id = st.session_state.get("dataset_id")
columns = st.session_state.get("columns")

if not dataset_id or not columns:
    st.error("🚫 No dataset loaded. Upload a dataset first.")
    st.stop()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
    <div style="padding:2rem;border-radius:14px;
                background:linear-gradient(135deg,#1e3c72,#2a5298);
                color:white">
        <h1>🤖 Enterprise AutoML Training</h1>
        <p>Reproducible • Explainable • Production-ready</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =====================================================
# TARGET SELECTION (MANDATORY)
# =====================================================
st.subheader("🎯 Target Variable")

target = st.selectbox(
    "Select target column",
    options=columns,
    help="Explicit target selection is mandatory in production ML"
)

# =====================================================
# TRAIN ACTION
# =====================================================
left, right = st.columns([3, 2])

with left:
    if st.button("🚀 Train Model", type="primary", use_container_width=True):
        with st.spinner("Training model…"):
            resp = requests.post(
                f"{API}/train/{dataset_id}",
                json={"target": target},
                timeout=300
            )

        if resp.status_code != 200:
            st.error(f"Backend error:\n{resp.text}")
            st.stop()

        result = resp.json()

        if result.get("status") != "ok":
            st.error("Training failed")
            st.json(result)
            st.stop()

        st.session_state.model_meta = result
        st.success("✅ Training completed successfully")
        st.rerun()

with right:
    st.info(
        """
        **Training pipeline**
        • Deterministic preprocessing  
        • Robust ensemble model  
        • Versioned artifacts  
        • Auditable metadata  
        """
    )

# =====================================================
# LOAD TRAINING RESULTS
# =====================================================
model_meta = st.session_state.get("model_meta")
if not model_meta:
    st.warning("No trained model yet.")
    st.stop()

# =====================================================
# SUMMARY
# =====================================================
st.markdown("## 🧠 Training Summary")

m1, m2, m3 = st.columns(3)
m1.metric("Task", model_meta["task"].upper())
m2.metric("Best Model", model_meta["best_model"])
m3.metric("Best Score", f"{model_meta['best_score']:.4f}")

# =====================================================
# LEADERBOARD (SAFE)
# =====================================================
st.markdown("## 🏆 Model Leaderboard")

leaderboard = model_meta.get("leaderboard", {})
rows = []

for model, score in leaderboard.items():
    if isinstance(score, (int, float)):
        rows.append({"Model": model, "Score": float(score)})

if not rows:
    st.warning("No valid leaderboard data.")
    st.json(leaderboard)
    st.stop()

df = pd.DataFrame(rows).sort_values("Score", ascending=False)

st.dataframe(
    df.style.format({"Score": "{:.4f}"})
      .background_gradient(subset=["Score"], cmap="Blues"),
    use_container_width=True
)

# =====================================================
# VISUAL COMPARISON
# =====================================================
st.markdown("## 📊 Performance Comparison")

fig = px.bar(
    df,
    x="Model",
    y="Score",
    text="Score",
    color="Model"
)
fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig.update_layout(height=420)

st.plotly_chart(fig, use_container_width=True)

# =====================================================
# MODEL SELECTION RATIONALE
# =====================================================
best = df.iloc[0]

st.markdown("## 🥇 Champion Model Rationale")
st.success(
    f"""
    **{best['Model']}** was selected because it achieved the highest
    validation score (**{best['Score']:.4f}**).

    **Decision criteria**
    • Objective metric optimization  
    • Stable preprocessing pipeline  
    • Ensemble robustness  
    """
)

# =====================================================
# EXPORT
# =====================================================
st.markdown("## 📥 Export Artifacts")

st.download_button(
    "Download Training Metadata (JSON)",
    data=json.dumps(model_meta, indent=2),
    file_name=f"training_report_{dataset_id}.json",
    mime="application/json",
    use_container_width=True
)

st.caption("Enterprise AutoML • Versioned • Auditable • Production-ready")
