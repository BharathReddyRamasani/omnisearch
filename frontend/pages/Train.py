import streamlit as st
import requests

st.markdown("""
<div style='background: linear-gradient(135deg, #2c5aa0 0%, #1e3c72 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
    <h1 style='font-size: 3rem;'>🤖 <b>AutoML Training</b></h1>
    <p style='font-size: 1.2rem;'>User-Controlled Target Selection</p>
</div>
""", unsafe_allow_html=True)

if "dataset_id" not in st.session_state:
    st.error("🚫 **Upload dataset first!**")
    st.stop()

dataset_id = st.session_state.dataset_id

# ✅ GET COLUMN NAMES FROM DATASET
st.markdown("### 🎯 **Select Target Column**")
try:
    resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}")
    data = resp.json()
    if data.get("status") == "ok":
        columns = list(data["eda"]["missing"].keys())  # All columns
        numeric_columns = [col for col in columns if data["eda"]["missing"][col] < 100]  # Low missing
        
        selected_target = st.selectbox(
            "🎯 **Choose target to predict:**",
            options=numeric_columns,
            index=0,
            help="Model will predict this column"
        )
    else:
        selected_target = st.text_input("Target column name")
except:
    selected_target = st.text_input("Target column name")

st.markdown("### 🚀 **Train Model**")
col1, col2 = st.columns(2)

with col1:
    st.info(f"**Training:** `{selected_target}`")
    st.markdown("- 🎯 User-selected target")
    st.markdown("- 🛠️ Auto-preprocessing") 
    st.markdown("- ⚡ Full pipeline")

if st.button(f"🚀 **Train Model → Predict {selected_target}**", type="primary", use_container_width=True):
    with st.spinner("🔬 Training production model..."):
        resp = requests.post(f"http://127.0.0.1:8000/train/{dataset_id}", 
                           json={"target": selected_target})
        data = resp.json()
        
        if data.get("status") == "ok":
            st.session_state.model_data = data
            st.session_state.model_trained = True
            st.session_state.selected_target = selected_target
            st.success(f"✅ **{data['message']}** 🎉")
            st.balloons()
            st.rerun()
        else:
            st.error(f"❌ **Training failed**: {data.get('message')}")

# RESULTS
if st.session_state.get("model_trained"):
    model_data = st.session_state.model_data
    
    st.markdown("### 📊 **Model Performance**")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("🎯 Target", model_data.get('target', 'N/A'))
    with col2: st.metric("📈 Score", f"{model_data.get('score', 0):.3f}")
    with col3: st.metric("📝 Task", model_data.get('task', '?').title())
    with col4: st.metric("🔧 Features", model_data.get('features_used', 0))
    
    st.success("✅ **Ready for Live Predictions!**")
    st.markdown("👆 **Next**: Go to **Predict** page")
