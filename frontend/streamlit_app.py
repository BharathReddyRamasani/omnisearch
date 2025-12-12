import streamlit as st
import requests
import base64

API_URL = "http://127.0.0.1:8000"

st.title("OmniSearch — CSV / Excel Workbench")

# ------------------------------------------------------
# PREVENT 'data' NOT DEFINED
# ------------------------------------------------------
data = None

# ------------------------------------------------------
# FILE UPLOAD
# ------------------------------------------------------
uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")
        }
        resp = requests.post(f"{API_URL}/upload", files=files)

        if resp.status_code == 200:
            data = resp.json()

            st.success(f"Uploaded Successfully: {data['filename']}")
            st.write("Detected Encoding:", data["encoding"])

            st.subheader("Column Mapping")
            for col in data["columns"]:
                st.write(f"{col['original']} → {col['normalized']}")

            st.subheader("Preview (first 5 rows)")
            st.dataframe(data["preview"])

        else:
            st.error(f"Upload failed: {resp.text}")

    except Exception as e:
        st.error(f"Server unreachable or crashed: {e}")


# ------------------------------------------------------
# RUN EDA
# ------------------------------------------------------
if data and st.button("Run EDA"):
    try:
        payload = {"rows": data["preview"]}  # send 5-row preview to backend
        eda_resp = requests.post(f"{API_URL}/profile", json=payload)

        if eda_resp.status_code == 200:
            eda = eda_resp.json()

            # MISSING VALUES
            st.subheader("Missing Values")
            st.write(eda.get("missing", {}))

            # DATA TYPES
            st.subheader("Data Types")
            st.write(eda.get("dtypes", {}))

            # STATS
            st.subheader("Summary Statistics")
            st.write(eda.get("stats", {}))

            # HISTOGRAMS
            st.subheader("Histograms")
            hists = eda.get("hists", {}) or {}
            for col, img_b64 in hists.items():
                if img_b64:
                    st.markdown(f"### {col}")
                    try:
                        img_bytes = base64.b64decode(img_b64)
                        st.image(img_bytes, width=700)
                    except Exception as e:
                        st.warning(f"Could not render histogram for {col}: {e}")

            # CORRELATION HEATMAP
            corr_b64 = eda.get("corr")
            if corr_b64:
                st.subheader("Correlation Heatmap")
                try:
                    corr_bytes = base64.b64decode(corr_b64)
                    st.image(corr_bytes, width=800)
                except Exception as e:
                    st.warning(f"Could not render heatmap: {e}")

        else:
            st.error(f"EDA failed: {eda_resp.text}")

    except Exception as e:
        st.error(f"EDA failed: {e}")

# ------------------------------------------------------
# END OF FILE
# ------------------------------------------------------
