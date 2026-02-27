# OmniSearch AI – Enterprise AutoML Dashboard 🚀

**One-line purpose:** An industrial-grade, end-to-end Machine Learning pipeline platform supporting robust CSV ingestion, intelligent data cleaning, advanced Exploratory Data Analysis (EDA), automated model training (AutoML), and live inference dashboards.

---

## 🌟 Project Overview

OmniSearch AI bridges the gap between raw data and predictive insights. It features a high-performance **FastAPI backend** handling heavy ML computation and a beautiful, interactive **Streamlit frontend** for user interaction and visualization. 

The system is designed to handle messy, real-world data with industrial-grade resilience, offering customizable ETL processes, unsupervised learning tools, and a suite of 14+ classification and regression algorithms.

---

## ✨ Core Features

### 1️⃣ Enterprise Data Ingestion ([Upload.py](cci:7://file:///d:/omnisearch/frontend/pages/Upload.py:0:0-0:0))
- **Encoding Detection:** Uses `charset-normalizer` to automatically detect and fallback decode files (UTF-8, cp1252, ISO-8859-1, etc.).
- **Smart Type Coercion:** Automatically infers and coerces numeric and datetime columns.
- **Column Normalization:** Cleans column names to snake_case and asks for user confirmation to maintain data governance.
- **Constraint Validation:** Handles files up to 500MB, 100k rows, and 500 columns safely.

### 2️⃣ Intelligent ETL & Cleaning ([ETL.py](cci:7://file:///d:/omnisearch/frontend/pages/ETL.py:0:0-0:0))
- Customizable cleaning strategies (Mean/Median imputation, Mode/Unknown for categoricals).
- **Outlier Treatment:** Configurable IQR or Z-Score based clipping/removal.
- **Date Handling:** Automatically infers date formats and normalizes decimal separators.
- **Quality Scoring:** Computes a 0-100 data quality score before and after cleaning to show "Expected Lift".
- **Reproducibility:** Saves cleaning configurations for future pipeline runs.

### 3️⃣ Advanced EDA Engine ([EDA.py](cci:7://file:///d:/omnisearch/frontend/pages/EDA.py:0:0-0:0) & [Dashboard.py](cci:7://file:///d:/omnisearch/frontend/pages/Dashboard.py:0:0-0:0))
- **Executive Dashboard:** Real-time KPIs, Missing Value Analysis, and Pipeline Completion tracking.
- **Statistical Profiling:** Detailed summary statistics, distribution analysis, and correlation heatmaps.
- **Dimensionality Reduction:** Built-in PCA and t-SNE projections.
- **Unsupervised Learning:** Auto-clustering (KMeans, DBSCAN, Agglomerative, Spectral) on selected features.
- **Anomaly Detection:** Identify outliers using Isolation Forest, One-Class SVM, or Local Outlier Factor.
- **Exportable Reports:** Download comprehensive JSON and HTML executive reports.

### 4️⃣ Industrial AutoML Training ([Train.py](cci:7://file:///d:/omnisearch/frontend/pages/Train.py:0:0-0:0))
- **Algorithm Arsenal:** Trains across Linear/Logistic Regression, Random Forests, Gradient Boosting, SVMs, KNN, Naive Bayes, etc.
- **Async Background Jobs:** Training runs in the background without blocking the UI.
- **Smart Defaults:** Auto-detects Classification vs. Regression based on target cardinality.
- **Advanced ML Features:** 
  - Learning Curves for Bias/Variance diagnostics.
  - Probability Calibration (Brier score) and Reliability Curves.
  - Model Complexity Metrics (Parameters, Model Size, Inference Latency).
  - Feature Importance calculation.
- **Experiment Tracking:** Leaderboard ranking, model history, and one-click model rollback.

### 5️⃣ Batch & Live Inference ([Predict.py](cci:7://file:///d:/omnisearch/frontend/pages/Predict.py:0:0-0:0))
- **Live Prediction:** Generates dynamic input forms based on dataset schema, filling in logical defaults (mean/mode).
- **Batch Prediction:** Upload a CSV for bulk inference with downloadable results.
- **Seamless Model Loading:** Automatically uses the latest Champion Model for predictions.

---

## 🛠️ System Architecture

The project uses a decoupled architecture for maximum scalability:

```text
Frontend (Streamlit)
  │   ├── app.py (Main entry point)
  │   ├── pages/ (Dashboard, Upload, EDA, ETL, Train, Predict)
  │   └── theme.py  (Custom CSS / Glassmorphism UI)
  ▼
Backend (FastAPI)  [http://127.0.0.1:8001/api]
  │   ├── app.py (Routing & Middleware)
  │   └── services/ 
  │       ├── ingest.py (File validation)
  │       ├── cleaning.py (ETL Logic)
  │       ├── eda.py, anomaly.py, cluster.py, advanced.py
  │       ├── training.py (Model Training)
  │       ├── predict.py (Inference Engine)
  │       └── model_registry.py (Version Control & History)
  ▼
Local Storage Layer
  ├── data/datasets/{dataset_id}/ (raw.csv, clean.csv, metadata.json)
  └── models/{dataset_id}/ (model.pkl, registry JSONs)

🚀 Installation & Running Locally
1. Requirements
Python 3.9+
pip or conda
2. Setup Virtual Environment
bash
# Create and activate environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Run the Application (Requires 2 Terminal Windows)
Terminal 1 (Backend - FastAPI)

bash
# From the root directory
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8001 --reload
Terminal 2 (Frontend - Streamlit)

bash
# Wait for backend to start, then run:
cd frontend
streamlit run app.py
The dashboard will be available at http://localhost:8501.

🛡️ Production Deployment Checklist
For production, do not run both servers on Streamlit Community Cloud.

Backend: Deploy backend/ to a VPS, Render, AWS, or Heroku. Ensure minimum 2GB RAM for ML operations.
Frontend: Deploy frontend/ to Streamlit Cloud or Vercel.
Configuration: Update the API = "http://127.0.0.1:8001/api" variable in the frontend pages to point to your live backend URL.
Database: (Optional Integration) Standardize the storage layer to use S3/GCS instead of local disk for stateless container scaling.
Built for scale, designed for simplicity.
