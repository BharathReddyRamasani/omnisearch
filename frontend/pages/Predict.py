# # import streamlit as st
# # import requests
# # import pandas as pd

# # st.markdown("""
# # <div style='background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
# #     <h1 style='font-size: 3rem;'>🔮 <b>Live Predictions</b></h1>
# #     <p style='font-size: 1.2rem;'>Production Model Inference Engine</p>
# # </div>
# # """, unsafe_allow_html=True)

# # if "dataset_id" not in st.session_state:
# #     st.error("🚫 **Upload dataset first!**")
# #     st.stop()

# # dataset_id = st.session_state.dataset_id

# # # ✅ FIXED MODEL CHECK
# # st.markdown("### 📊 **Model Status**")
# # try:
# #     resp = requests.get(f"http://127.0.0.1:8000/meta/{dataset_id}", timeout=5)
# #     meta = resp.json()
    
# #     if meta.get("status") == "ok":
# #         st.success(f"✅ **Model Ready!** Target: `{meta['target']}` | Score: {meta.get('score', 0):.3f}")
# #         model_ready = True
# #         st.session_state.model_meta = meta
# #     else:
# #         st.error(f"❌ **{meta.get('message', 'Model not ready')}**")
# #         st.info("👆 Go to **Train** page → Select target → Train model")
# #         model_ready = False
# # except:
# #     st.error("⚠️ **Backend connection failed**")
# #     model_ready = False

# # if not model_ready:
# #     st.stop()

# # # ✅ DYNAMIC PREDICTION FORM
# # st.markdown("### 📝 **Live Prediction Input**")
# # meta = st.session_state.model_meta

# # # Show target being predicted
# # st.info(f"**Predicting:** `{meta['target']}`")

# # # Simple input form (5 generic features)
# # col1, col2 = st.columns(2)
# # input_data = {}

# # input_data["feature1"] = col1.number_input("Feature 1", value=0.0, step=0.1)
# # input_data["feature2"] = col2.number_input("Feature 2", value=0.0, step=0.1)
# # input_data["feature3"] = st.number_input("Feature 3", value=0.0, step=0.1)
# # input_data["feature4"] = st.number_input("Feature 4", value=0.0, step=0.1)
# # input_data["feature5"] = st.number_input("Feature 5", value=0.0, step=0.1)

# # if st.button("🔮 **Run Production Prediction**", type="primary", use_container_width=True):
# #     with st.spinner("⚡ Live inference..."):
# #         resp = requests.post(
# #             f"http://127.0.0.1:8000/predict/{dataset_id}",
# #             json={"input_data": input_data},
# #             timeout=10
# #         )
# #         result = resp.json()
        
# #         if result.get("status") == "ok":
# #             st.markdown("### 🎯 **Production Prediction Result**")
# #             col1, col2 = st.columns(2)
# #             with col1:
# #                 st.metric(f"**Predicted {meta['target']}**", f"{result['prediction']:.3f}")
# #             with col2:
# #                 st.success("✅ **Live inference successful!**")
# #             st.balloons()
# #         else:
# #             st.error(f"❌ **Prediction failed**: {result.get('message')}")

# # st.markdown("---")
# # st.success("✅ **Full ML Pipeline: Upload → EDA → Train → Predict**")

# # # frontend/Predict.py - QUICK COPY
# # st.title("🔮 Live Predictions")
# # if st.session_state.get("model_trained"):
# #     meta = requests.get(f"http://127.0.0.1:8000/meta/{dataset_id}").json()
# #     st.metric("🏆 Best Model", meta.get("best_model"))
    
# #     # Dynamic form from features
# #     input_data = {}
# #     for feature in meta.get("features", [])[:5]:  # Top 5
# #         input_data[feature] = st.number_input(feature)
    
# #     if st.button("🔮 Predict"):
# #         resp = requests.post(f"http://127.0.0.1:8000/predict/{dataset_id}", 
# #                            json={"input_data": input_data})
# #         st.success(f"🎯 **{meta['target']}:** {resp.json()['prediction']}")

# import streamlit as st
# import requests
# import pandas as pd

# API = "http://127.0.0.1:8000/api"

# st.set_page_config(
#     page_title="OmniSearch AI – Prediction",
#     layout="wide",
# )

# # =====================================================
# # PRECHECKS
# # =====================================================
# dataset_id = st.session_state.get("dataset_id")
# model_meta = st.session_state.get("model_meta")

# if not dataset_id or not model_meta:
#     st.error("🚫 Train a model before prediction.")
#     st.stop()

# # =====================================================
# # HEADER
# # =====================================================
# st.markdown(
#     """
# <div style="background:linear-gradient(90deg,#141e30,#243b55);
# padding:2rem;border-radius:16px;color:white;">
# <h1 style="margin:0;">🎯 Enterprise Prediction Engine</h1>
# <p style="opacity:.9;">Reliable • Explainable • Production-Ready</p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# # =====================================================
# # MODE SELECTION
# # =====================================================
# st.markdown("## 🔀 Prediction Mode")

# mode = st.radio(
#     "",
#     ["Smart Mode (Top Impact Features)", "Full Mode (All Features)"],
#     horizontal=True,
# )

# use_top = "Smart" in mode

# features = (
#     model_meta["top_features"]
#     if use_top
#     else list(model_meta["feature_defaults"].keys())
# )

# defaults = model_meta["feature_defaults"]

# st.info(
#     "Smart Mode uses **only the most impactful features**.\n"
#     "Missing values are auto-filled using training statistics."
# )

# # =====================================================
# # INPUT FORM
# # =====================================================
# st.markdown("## 🧾 Input Features")

# payload = {}
# cols = st.columns(3)

# for i, f in enumerate(features):
#     payload[f] = cols[i % 3].text_input(
#         f,
#         placeholder=f"Default → {defaults[f]}"
#     )

# payload["_mode"] = "top" if use_top else "full"

# # =====================================================
# # PREDICT
# # =====================================================
# st.markdown("---")
# if st.button("🚀 Run Prediction", type="primary"):
#     with st.spinner("Scoring model…"):
#         r = requests.post(
#             f"{API}/predict/{dataset_id}",
#             json=payload,
#             timeout=30
#         )

#     res = r.json()

#     if res.get("status") != "ok":
#         st.error(res.get("error", "Prediction failed"))
#         st.stop()

#     st.success("Prediction Successful")

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Prediction", res["prediction"])
#     if res.get("confidence") is not None:
#         c2.metric("Confidence", f"{res['confidence']:.2f}")
#     c3.metric("Mode", res["mode"].upper())

#     # =====================================================
#     # TRANSPARENCY PANEL
#     # =====================================================
#     with st.expander("🔍 Prediction Transparency"):
#         st.markdown("**Features Used:**")
#         st.write(res["used_features"])

#         if res["auto_filled"]:
#             st.warning(
#                 "Auto-filled fields:\n" + ", ".join(res["auto_filled"])
#             )
#         else:
#             st.success("No auto-filled fields.")


import streamlit as st
import requests

API = "http://127.0.0.1:8000/api"

st.set_page_config(page_title="OmniSearch AI – Predict", layout="wide")

dataset_id = st.session_state.get("dataset_id")
model_meta = st.session_state.get("model_meta")

if not dataset_id or not model_meta:
    st.error("🚫 Train a model before prediction.")
    st.stop()

st.markdown("## 🎯 Enterprise Prediction Engine")

mode = st.radio(
    "Prediction Mode",
    ["Smart Mode (Top Impact Features)", "Full Mode (All Features)"],
    horizontal=True,
)

use_top = "Smart" in mode
features = model_meta["top_features"] if use_top else model_meta["raw_columns"]
defaults = model_meta["feature_defaults"]

payload = {}
st.markdown("### Input Features")

cols = st.columns(3)
for i, f in enumerate(features):
    with cols[i % 3]:
        payload[f] = st.text_input(
            label=f,
            value=str(defaults.get(f, "")),
        )

payload["_mode"] = "top" if use_top else "full"

if st.button("🚀 Predict", type="primary"):
    with st.spinner("Scoring model…"):
        r = requests.post(f"{API}/predict/{dataset_id}", json=payload)
        res = r.json()

    if res.get("status") != "ok":
        st.error(res.get("error", "Prediction failed"))
        st.stop()

    st.success("Prediction successful")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", res["prediction"])
    if res.get("confidence") is not None:
        c2.metric("Confidence", f"{res['confidence']*100:.1f}%")
    c3.metric("Mode", res["mode"].upper())

    with st.expander("🔍 Transparency"):
        st.write("**Used Features:**", res["used_features"])
        if res["auto_filled"]:
            st.warning("Auto-filled: " + ", ".join(res["auto_filled"]))
        else:
            st.success("No auto-filled fields")
