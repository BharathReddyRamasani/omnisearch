# # # pages/EDA.py – FINAL VERSION

# # import streamlit as st
# # import requests
# # import pandas as pd
# # import plotly.express as px
# # import plotly.graph_objects as go
# # import json
# # from datetime import datetime

# # st.set_page_config(
# #     page_title="OmniSearch AI - Enterprise EDA",
# #     layout="wide",
# #     initial_sidebar_state="expanded",
# # )

# # # ---------- DATASET CHECK ----------
# # if "dataset_id" not in st.session_state:
# #     st.error("🚫 **No dataset loaded**. Go to **Upload** page first.")
# #     st.stop()

# # dataset_id = st.session_state.dataset_id

# # # ---------- HEADER ----------
# # st.markdown(
# #     """
# # <div style='background: linear-gradient(90deg, #1e3a72 0%, #2c5aa0 100%); padding: 2.5rem; border-radius: 18px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
# #     <h1 style='margin: 0; font-size: 3.5rem;'>📊 <b>Enterprise EDA Engine</b></h1>
# #     <p style='margin: 8px 0 0 0; font-size: 1.4rem; opacity: 0.9;'>ETL Intelligence • Production Analytics • Industrial Quality Scoring</p>
# # </div>
# # """,
# #     unsafe_allow_html=True,
# # )

# # # ---------- SIDEBAR ----------
# # with st.sidebar:
# #     st.markdown("### 🎛️ **EDA Controls**")
# #     if st.button("🔄 **Refresh Analysis**", use_container_width=True, type="secondary"):
# #         # Clear EDA state and show only usage text
# #         for key in list(st.session_state.keys()):
# #             if key in ["eda_data", "eda_status"] or key.startswith("eda_"):
# #                 del st.session_state[key]
# #         st.rerun()

# #     st.markdown("---")
# #     st.markdown("### 📊 **Quick Status**")
# #     if st.session_state.get("eda_data"):
# #         quick = st.session_state.eda_data
# #         st.metric("Rows", f"{quick.get('rows', 0):,}")
# #         st.metric("Quality", f"{quick.get('quality_score', 0):.0f}%")
# #         st.metric("Source", quick.get("data_source", "RAW"))

# # # ---------- RUN EDA BUTTON ----------
# # run_clicked = st.button(
# #     "🚀 **Run Enterprise EDA Pipeline**",
# #     type="primary",
# #     use_container_width=True,
# # )

# # # Need a fresh run if:
# # # - no data yet, or
# # # - status not complete, or
# # # - user clicked Run button
# # need_run = (
# #     "eda_data" not in st.session_state
# #     or "eda_status" not in st.session_state
# #     or st.session_state.get("eda_status") != "complete"
# # )

# # if run_clicked or need_run:
# #     if run_clicked:  # only call backend when user explicitly clicks
# #         with st.spinner("🔬 Running EDA (backend chooses RAW vs CLEAN)..."):
# #             try:
# #                 resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}", timeout=60)
# #                 if resp.status_code != 200:
# #                     st.error(f"Backend error: {resp.status_code}")
# #                     st.stop()

# #                 data = resp.json()
# #                 if data.get("status") != "ok":
# #                     st.error(f"EDA failed: {data.get('message', 'Unknown error')}")
# #                     st.stop()

# #                 st.session_state.eda_data = data["eda"]
# #                 st.session_state.eda_status = "complete"
# #                 st.success("✅ **Enterprise EDA Complete!**")
# #                 st.rerun()
# #             except Exception as e:
# #                 st.error(f"Error contacting backend: {str(e)}")
# #                 st.stop()
# #     else:
# #         # No EDA data and user has not clicked yet: show usage text only
# #         st.markdown(
# #             """
# # ### 🚀 Enterprise EDA Pipeline Ready

# # **What this page does:**

# # - If ETL **not** run → analyzes the **raw file**.
# # - If you clicked **ETL Clean Data** → analyzes the **cleaned file** (`clean.csv`).
# # - Computes:
# #   - Missing value profile  
# #   - Quality score (0–100)  
# #   - Summary statistics  
# #   - ETL impact (outliers & missing fixed, if available)

# # Click **"Run Enterprise EDA Pipeline"** above to start.
# # """
# #         )
# #         st.stop()

# # # ---------- DISPLAY RESULTS ----------
# # eda = st.session_state.get("eda_data")
# # if not eda:
# #     st.stop()

# # # 1. Executive Summary
# # st.markdown("### 🎯 **Executive Summary Dashboard**")
# # c1, c2, c3, c4, c5, c6 = st.columns(6)
# # with c1:
# #     st.metric("📊 Rows", f"{eda['rows']:,}")
# # with c2:
# #     st.metric("📋 Columns", eda["columns"])
# # with c3:
# #     st.metric("🚫 Missing %", f"{eda['missing_pct']:.1f}%")
# # with c4:
# #     st.metric("⚡ Quality", f"{eda['quality_score']:.0f}%")
# # with c5:
# #     st.metric("📈 Source", eda.get("data_source", "RAW"))
# # with c6:
# #     st.metric("🧹 ETL", "✅ CLEAN" if eda.get("etl_complete") else "RAW")

# # # 2. ETL Intelligence
# # st.markdown("### 🧹 **ETL Intelligence**")
# # if eda.get("etl_complete") and eda.get("etl_improvements"):
# #     etl_imp = eda["etl_improvements"]
# #     imp = etl_imp["improvements"]
# #     lift = etl_imp["accuracy_lift_expected"]

# #     c1, c2, c3 = st.columns(3)
# #     with c1:
# #         st.metric("⚠️ Outliers FIXED", f"{imp['outliers_fixed']:,}")
# #     with c2:
# #         st.metric("🔢 Missing FILLED", f"{imp['missing_values_filled']:,}")
# #     with c3:
# #         st.metric("🎯 Expected Lift", f"+{lift:.1f}%")

# #     st.success(
# #         f"✅ ETL Impact: {imp['outliers_fixed']:,} outliers and "
# #         f"{imp['missing_values_filled']:,} missing values fixed."
# #     )
# # else:
# #     st.info("ℹ️ ETL not run yet – EDA currently using RAW data.")

# # # 3. Quality Gauge
# # st.markdown("### ⚡ **Data Quality Gauge**")
# # fig_gauge = go.Figure(
# #     go.Indicator(
# #         mode="gauge+number",
# #         value=eda["quality_score"],
# #         domain={"x": [0, 1], "y": [0, 1]},
# #         title={"text": "Data Quality Score"},
# #         gauge={
# #             "axis": {"range": [0, 100]},
# #             "bar": {"color": "#2c5aa0"},
# #             "steps": [
# #                 {"range": [0, 70], "color": "red"},
# #                 {"range": [70, 90], "color": "orange"},
# #                 {"range": [90, 100], "color": "green"},
# #             ],
# #         },
# #     )
# # )
# # st.plotly_chart(fig_gauge, use_container_width=True)

# # # 4. Missing Value Intelligence
# # st.markdown("### 🚫 **Missing Value Intelligence**")
# # if eda.get("missing"):
# #     missing_df = pd.DataFrame(list(eda["missing"].items()), columns=["Column", "Count"])
# #     missing_df["%"] = (missing_df["Count"] / eda["rows"] * 100).round(2)
# #     missing_df = missing_df.sort_values("%", ascending=False).head(15)

# #     fig_missing = px.bar(
# #         missing_df,
# #         x="Column",
# #         y="%",
# #         color="%",
# #         color_continuous_scale="Reds",
# #         title="Top 15 Columns by Missing %",
# #     )
# #     fig_missing.update_layout(height=400, xaxis_tickangle=-45)
# #     st.plotly_chart(fig_missing, use_container_width=True)
# # else:
# #     st.success("✅ Zero missing values detected!")

# # # 5. Statistical Profile
# # st.markdown("### 📈 **Statistical Profile**")
# # if eda.get("summary"):
# #     summary_df = pd.DataFrame(eda["summary"]).T.round(2)
# #     st.dataframe(summary_df, use_container_width=True)

# # # 6. Exports
# # st.markdown("### 📥 **Exports**")
# # cc1, cc2 = st.columns(2)
# # with cc1:
# #     st.download_button(
# #         "📄 Full JSON Report",
# #         data=json.dumps(eda, indent=2),
# #         file_name=f"EDA_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
# #         mime="application/json",
# #     )
# # with cc2:
# #     st.download_button(
# #         "📊 Summary CSV",
# #         data=pd.DataFrame([eda]).to_csv(index=False),
# #         file_name=f"EDA_Summary_{dataset_id}.csv",
# #     )

# # pages/EDA.py – FINAL BEHAVIOUR: refresh clears, run uses raw/clean correctly

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import json
# from datetime import datetime

# st.set_page_config(
#     page_title="OmniSearch AI - Enterprise EDA",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ---------- DATASET CHECK ----------
# if "dataset_id" not in st.session_state:
#     st.error("🚫 No dataset loaded. Go to Upload page first.")
#     st.stop()

# dataset_id = st.session_state.dataset_id

# # ---------- HEADER ----------
# st.markdown(
#     """
# <div style='background: linear-gradient(90deg, #1e3a72 0%, #2c5aa0 100%); padding: 2.5rem; border-radius: 18px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
#     <h1 style='margin: 0; font-size: 3.5rem;'>📊 <b>Enterprise EDA Engine</b></h1>
#     <p style='margin: 8px 0 0 0; font-size: 1.4rem; opacity: 0.9;'>ETL Intelligence • Production Analytics • Industrial Quality Scoring</p>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# # ---------- SIDEBAR ----------
# with st.sidebar:
#     st.markdown("### 🎛️ EDA Controls")
#     if st.button("🔄 Refresh Analysis", use_container_width=True):
#         # clear all EDA state – after this only usage text should appear
#         for key in list(st.session_state.keys()):
#             if key in ["eda_data", "eda_status"] or key.startswith("eda_"):
#                 del st.session_state[key]
#         st.rerun()

#     st.markdown("---")
#     st.markdown("### 📊 Quick Status")
#     if st.session_state.get("eda_data"):
#         quick = st.session_state.eda_data
#         st.metric("Rows", f"{quick.get('rows', 0):,}")
#         st.metric("Quality", f"{quick.get('quality_score', 0):.0f}%")
#         st.metric("Source", quick.get("data_source", "RAW"))

# # ---------- RUN BUTTON ----------
# run_clicked = st.button(
#     "🚀 Run Enterprise EDA Pipeline",
#     type="primary",
#     use_container_width=True,
# )

# # If no EDA data and user hasn't clicked Run yet → show only usage text
# if "eda_data" not in st.session_state and not run_clicked:
#     st.markdown(
#         """
# ### 🚀 EDA Ready – How it behaves

# - If you **have NOT** run ETL Clean yet → EDA analyzes the original **RAW file**.
# - If you **clicked ETL Clean Data** in the ETL page → EDA analyzes the **CLEANED file** (`clean.csv`).

# What you get when you run:

# - Data health (missing %, quality score)
# - Distributions and basic statistics
# - ETL impact (outliers / missing fixed) when clean data exists

# Click **"Run Enterprise EDA Pipeline"** above to start.
# """
#     )
#     st.stop()

# # ---------- CALL BACKEND WHEN RUN CLICKED ----------
# if run_clicked:
#     with st.spinner("🔬 Running EDA (backend chooses RAW vs CLEAN)..."):
#         try:
#             resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}", timeout=60)
#             if resp.status_code != 200:
#                 st.error(f"Backend error: {resp.status_code}")
#                 st.stop()

#             data = resp.json()
#             if data.get("status") != "ok":
#                 st.error(f"EDA failed: {data.get('message', 'Unknown error')}")
#                 st.stop()

#             st.session_state.eda_data = data["eda"]
#             st.session_state.eda_status = "complete"
#             st.success("✅ EDA complete.")
#             st.rerun()
#         except Exception as e:
#             st.error(f"Error contacting backend: {str(e)}")
#             st.stop()

# # ---------- SHOW RESULTS ----------
# eda = st.session_state.get("eda_data")
# if not eda:
#     st.stop()

# st.markdown("### 🎯 Executive Summary Dashboard")
# c1, c2, c3, c4, c5, c6 = st.columns(6)
# with c1:
#     st.metric("Rows", f"{eda['rows']:,}")
# with c2:
#     st.metric("Columns", eda["columns"])
# with c3:
#     st.metric("Missing %", f"{eda['missing_pct']:.1f}%")
# with c4:
#     st.metric("Quality", f"{eda['quality_score']:.0f}%")
# with c5:
#     st.metric("Source", eda.get("data_source", "RAW"))
# with c6:
#     st.metric("ETL", "CLEAN" if eda.get("etl_complete") else "RAW")

# st.markdown("### 🧹 ETL Intelligence")
# if eda.get("etl_complete") and eda.get("etl_improvements"):
#     imp_block = eda["etl_improvements"]["improvements"]
#     lift = eda["etl_improvements"]["accuracy_lift_expected"]
#     c1, c2, c3 = st.columns(3)
#     with c1:
#         st.metric("Outliers fixed", f"{imp_block['outliers_fixed']:,}")
#     with c2:
#         st.metric("Missing filled", f"{imp_block['missing_values_filled']:,}")
#     with c3:
#         st.metric("Expected lift", f"+{lift:.1f}%")
# else:
#     st.info("ETL not run yet – this EDA uses RAW data only.")

# st.markdown("### ⚡ Data Quality Gauge")
# fig_gauge = go.Figure(
#     go.Indicator(
#         mode="gauge+number",
#         value=eda["quality_score"],
#         domain={"x": [0, 1], "y": [0, 1]},
#         title={"text": "Data Quality Score"},
#         gauge={
#             "axis": {"range": [0, 100]},
#             "bar": {"color": "#2c5aa0"},
#             "steps": [
#                 {"range": [0, 70], "color": "red"},
#                 {"range": [70, 90], "color": "orange"},
#                 {"range": [90, 100], "color": "green"},
#             ],
#         },
#     )
# )
# st.plotly_chart(fig_gauge, use_container_width=True)

# st.markdown("### 🚫 Missing Value Intelligence")
# if eda.get("missing"):
#     missing_df = pd.DataFrame(list(eda["missing"].items()), columns=["Column", "Count"])
#     missing_df["%"] = (missing_df["Count"] / eda["rows"] * 100).round(2)
#     missing_df = missing_df.sort_values("%", ascending=False).head(15)
#     fig_missing = px.bar(
#         missing_df,
#         x="Column",
#         y="%",
#         color="%",
#         color_continuous_scale="Reds",
#         title="Top 15 Columns by Missing %",
#     )
#     fig_missing.update_layout(height=400, xaxis_tickangle=-45)
#     st.plotly_chart(fig_missing, use_container_width=True)
# else:
#     st.success("Zero missing values detected.")

# st.markdown("### 📈 Statistical Profile")
# if eda.get("summary"):
#     summary_df = pd.DataFrame(eda["summary"]).T.round(2)
#     st.dataframe(summary_df, use_container_width=True)

# st.markdown("### 📥 Exports")
# c1, c2 = st.columns(2)
# with c1:
#     st.download_button(
#         "Download JSON Report",
#         data=json.dumps(eda, indent=2),
#         file_name=f"EDA_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
#         mime="application/json",
#     )
# with c2:
#     st.download_button(
#         "Download Summary CSV",
#         data=pd.DataFrame([eda]).to_csv(index=False),
#         file_name=f"EDA_Summary_{dataset_id}.csv",
#     )

# pages/EDA.py – FINAL BEHAVIOUR: refresh clears, run uses raw/clean correctly

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime

st.set_page_config(
    page_title="OmniSearch AI - Enterprise EDA",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DATASET CHECK ----------
if "dataset_id" not in st.session_state:
    st.error("🚫 No dataset loaded. Go to Upload page first.")
    st.stop()

dataset_id = st.session_state.dataset_id

# ---------- HEADER ----------
st.markdown(
    """
<div style='background: linear-gradient(90deg, #1e3a72 0%, #2c5aa0 100%); padding: 2.5rem; border-radius: 18px; color: white; text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.3);'>
    <h1 style='margin: 0; font-size: 3.5rem;'>📊 <b>Enterprise EDA Engine</b></h1>
    <p style='margin: 8px 0 0 0; font-size: 1.4rem; opacity: 0.9;'>ETL Intelligence • Production Analytics • Industrial Quality Scoring</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🎛️ EDA Controls")
    if st.button("🔄 Refresh Analysis", use_container_width=True):
        # clear all EDA state – after this only usage text should appear
        for key in list(st.session_state.keys()):
            if key in ["eda_data", "eda_status"] or key.startswith("eda_"):
                del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 Quick Status")
    if st.session_state.get("eda_data"):
        quick = st.session_state.eda_data
        st.metric("Rows", f"{quick.get('rows', 0):,}")
        st.metric("Quality", f"{quick.get('quality_score', 0):.0f}%")
        st.metric("Source", quick.get("data_source", "RAW"))

# ---------- RUN BUTTON ----------
run_clicked = st.button(
    "🚀 Run Enterprise EDA Pipeline",
    type="primary",
    use_container_width=True,
)

# If no EDA data and user hasn't clicked Run yet → show only usage text
if "eda_data" not in st.session_state and not run_clicked:
    st.markdown(
        """
### 🚀 EDA Ready – How it behaves

- If you **have NOT** run ETL Clean yet → EDA analyzes the original **RAW file**.
- If you **clicked ETL Clean Data** in the ETL page → EDA analyzes the **CLEANED file** (`clean.csv`).

What you get when you run:

- Data health (missing %, quality score)
- Distributions and basic statistics
- ETL impact (outliers / missing fixed) when clean data exists

Click **"Run Enterprise EDA Pipeline"** above to start.
"""
    )
    st.stop()

# ---------- CALL BACKEND WHEN RUN CLICKED ----------
if run_clicked:
    with st.spinner("🔬 Running EDA (backend chooses RAW vs CLEAN)..."):
        try:
            resp = requests.get(f"http://127.0.0.1:8000/eda/{dataset_id}", timeout=60)
            if resp.status_code != 200:
                st.error(f"Backend error: {resp.status_code}")
                st.stop()

            data = resp.json()
            if data.get("status") != "ok":
                st.error(f"EDA failed: {data.get('message', 'Unknown error')}")
                st.stop()

            st.session_state.eda_data = data["eda"]
            st.session_state.eda_status = "complete"
            st.success("✅ EDA complete.")
            st.rerun()
        except Exception as e:
            st.error(f"Error contacting backend: {str(e)}")
            st.stop()

# ---------- SHOW RESULTS ----------
eda = st.session_state.get("eda_data")
if not eda:
    st.stop()

st.markdown("### 🎯 Executive Summary Dashboard")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("Rows", f"{eda['rows']:,}")
with c2:
    st.metric("Columns", eda["columns"])
with c3:
    st.metric("Missing %", f"{eda['missing_pct']:.1f}%")
with c4:
    st.metric("Quality", f"{eda['quality_score']:.0f}%")
with c5:
    st.metric("Source", eda.get("data_source", "RAW"))
with c6:
    st.metric("ETL", "CLEAN" if eda.get("etl_complete") else "RAW")

st.markdown("### 🧹 ETL Intelligence")
if eda.get("etl_complete") and eda.get("etl_improvements"):
    imp_block = eda["etl_improvements"]["improvements"]
    lift = eda["etl_improvements"]["accuracy_lift_expected"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Outliers fixed", f"{imp_block['outliers_fixed']:,}")
    with c2:
        st.metric("Missing filled", f"{imp_block['missing_values_filled']:,}")
    with c3:
        st.metric("Expected lift", f"+{lift:.1f}%")
else:
    st.info("ETL not run yet – this EDA uses RAW data only.")

st.markdown("### ⚡ Data Quality Gauge")
fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=eda["quality_score"],
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Data Quality Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#2c5aa0"},
            "steps": [
                {"range": [0, 70], "color": "red"},
                {"range": [70, 90], "color": "orange"},
                {"range": [90, 100], "color": "green"},
            ],
        },
    )
)
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("### 🚫 Missing Value Intelligence")
if eda.get("missing"):
    missing_df = pd.DataFrame(list(eda["missing"].items()), columns=["Column", "Count"])
    missing_df["%"] = (missing_df["Count"] / eda["rows"] * 100).round(2)
    missing_df = missing_df.sort_values("%", ascending=False).head(15)
    fig_missing = px.bar(
        missing_df,
        x="Column",
        y="%",
        color="%",
        color_continuous_scale="Reds",
        title="Top 15 Columns by Missing %",
    )
    fig_missing.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_missing, use_container_width=True)
else:
    st.success("Zero missing values detected.")

st.markdown("### 📈 Statistical Profile")
if eda.get("summary"):
    summary_df = pd.DataFrame(eda["summary"]).T.round(2)
    st.dataframe(summary_df, use_container_width=True)

st.markdown("### 📥 Exports")
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download JSON Report",
        data=json.dumps(eda, indent=2),
        file_name=f"EDA_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json",
    )
with c2:
    st.download_button(
        "Download Summary CSV",
        data=pd.DataFrame([eda]).to_csv(index=False),
        file_name=f"EDA_Summary_{dataset_id}.csv",
    )
