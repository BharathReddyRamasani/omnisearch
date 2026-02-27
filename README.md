# OmniSearch AI 🚀

> **Industrial-Grade End-to-End Machine Learning Platform**  
> Upload any CSV → Auto EDA → Auto Clean → AutoML → Predictions → LLM-Powered Analytics

---

## 📌 Table of Contents

- [What Is OmniSearch AI?](#what-is-omnisearch-ai)
- [Real-World Use Cases](#real-world-use-cases)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Workflow Walkthrough](#workflow-walkthrough)
- [ML Algorithms Supported](#ml-algorithms-supported)
- [Advanced Features](#advanced-features)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Known Limitations](#known-limitations)
- [Future Roadmap](#future-roadmap)
- [Resume Bullets](#resume-bullets)

---

## What Is OmniSearch AI?

OmniSearch AI is an **end-to-end AI workbench** that bridges the gap between raw tabular data and predictive insights — without requiring any ML expertise from the user.

A user uploads a CSV file. The system does the rest:

- **Ingests** the file with encoding detection and schema inference
- **Profiles** the data with automated EDA (distributions, outliers, correlations)
- **Cleans** the data with configurable ETL strategies and quality scoring
- **Trains** ML models automatically across 14+ algorithms
- **Serves** predictions through a live inference interface
- **Explains** results through an LLM-powered DSL agent

This is not a CSV viewer. This is a **production-grade ML pipeline platform** with Glassmorphism UI, background training jobs, model versioning, anomaly detection, clustering, dimensionality reduction, and a safe LLM analytics layer.

---

## Real-World Use Cases

| User | Task | OmniSearch Does |
|------|------|-----------------|
| Product Manager | Upload sales CSV → "Why did revenue drop?" | EDA + Anomaly Detection + LLM explanation |
| Data Analyst | Train churn prediction model | AutoML → Best model → REST prediction endpoint |
| ML Engineer | Compare 14 algorithms on a dataset | Model leaderboard + metrics + feature importance |
| Business User | Predict price for new inputs | Dynamic form → live inference → confidence score |
| Developer | Integrate predictions into an app | JSON prediction API, auto-generated per dataset |

---

## Core Features

### 1. Enterprise Data Ingestion (`Upload.py`)

- **Encoding Detection** — Uses `charset-normalizer` to auto-detect and decode files (UTF-8, cp1252, ISO-8859-1, UTF-16, etc.)
- **Smart Type Coercion** — Automatically infers and coerces numeric and datetime columns
- **Column Normalization** — Cleans column names to `snake_case` with user confirmation
- **Constraint Validation** — Handles files up to 500MB, 100k rows, 500 columns
- **Schema Persistence** — Stores `schema.json` for downstream ML pipeline stages
- **Preview & Mapping** — Shows old→new column name mapping for data governance

### 2. Intelligent ETL & Cleaning (`ETL.py`)

- Configurable cleaning strategies: Mean/Median imputation for numerics, Mode/Unknown for categoricals
- **Outlier Treatment** — IQR or Z-Score based clipping/removal with configurable thresholds
- **Date Handling** — Heuristic date format detection + user override for ambiguous formats
- **Data Quality Scoring** — 0–100 quality score computed before and after cleaning to show "Expected Lift"
- **Reproducibility** — Saves cleaning configurations as part of the dataset metadata
- **Before/After Preview** — Visual diff of distributions pre and post-cleaning

### 3. Advanced EDA Engine (`EDA.py` + `Dashboard.py`)

- **Executive Dashboard** — Real-time KPIs, Missing Value Analysis, Pipeline Completion tracking
- **Statistical Profiling** — Summary statistics, distribution plots, correlation heatmaps
- **Dimensionality Reduction** — Built-in PCA and t-SNE projections for high-dimensional datasets
- **Unsupervised Clustering** — Auto-clustering using KMeans, DBSCAN, Agglomerative, Spectral on selected features
- **Anomaly Detection** — Detect outliers using Isolation Forest, One-Class SVM, or Local Outlier Factor
- **Exportable Reports** — Download comprehensive JSON and HTML executive reports
- All plots rendered as **base64-encoded images** — no filesystem leaks, safe for server-side rendering

### 4. Industrial AutoML Training (`Train.py`)

**14+ Algorithms across classification and regression:**

| Category | Algorithms |
|----------|-----------|
| Linear | Logistic Regression, Linear Regression, Ridge, Lasso, SGD |
| Ensemble | Random Forest, Gradient Boosting, HistGradientBoosting |
| Boosting | XGBoost (optional), LightGBM (optional) |
| Tree | Decision Tree |
| Probabilistic | Gaussian Naive Bayes |
| SVM | (via calibrated classifiers) |

**Advanced ML Pipeline Features:**
- **Background Training Jobs** — Training runs asynchronously via background threads; UI never freezes
- **Quick Sample Model** — 10-second preview model for instant feedback before full training
- **Auto Task Detection** — Classifies as Classification or Regression based on target cardinality
- **Strict Pipeline Hygiene** — Train/test split BEFORE imputation to prevent data leakage
- **Cross-Validation** — Stratified K-Fold (classification) / K-Fold (regression) with mean ± std metrics
- **Learning Curves** — Bias/variance diagnostics for model complexity analysis
- **Probability Calibration** — Brier score + reliability curves for classification models
- **Model Complexity Metrics** — Parameter count, model size, inference latency benchmarks
- **Feature Importance** — Extracted and stored per-target for downstream explainability
- **SMOTE** — Optional class imbalance handling via oversampling (when `imbalanced-learn` is available)
- **Experiment Tracking** — Full leaderboard with model history and one-click rollback via `ModelRegistry`

### 5. Batch & Live Inference (`Predict.py`)

- **Dynamic Input Forms** — Auto-generated from dataset schema; numeric fields, dropdowns for categoricals
- **Smart Defaults** — Missing inputs auto-filled with training-time median/mode values
- **Quick Predict Mode** — Shows only top-K feature inputs based on feature importance
- **Advanced Predict Mode** — Full feature input form for power users
- **Batch Prediction** — Upload a CSV for bulk inference; download results as CSV
- **Champion Model** — Always loads the latest active model from `ModelRegistry`
- **Confidence Scores** — Probability output for classification tasks

### 6. LLM-Powered Analytics (DSL Agent)

- **Zero Hallucination Design** — LLM outputs strict JSON DSL; server executes safe helper functions only
- **No Raw Code Execution** — All actions mapped to validated server-side executors (`execute_dsl`)
- **Supported DSL Actions** — `agg`, `groupby`, `filter`, `plot`, SQL-style queries
- **FAISS Vector Store** — Column descriptions and EDA summaries embedded and indexed for RAG-style retrieval
- **Schema-Grounded Prompts** — LLM receives column names + types only, never raw data rows
- **Planner + Validator + Explainer** — Modular LLM pipeline with independent validation layer

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit Frontend                     │
│  app.py → pages/: Upload, EDA, ETL, Train, Predict       │
│  Custom Glassmorphism UI via theme.py                    │
└────────────────────────┬─────────────────────────────────┘
                         │  REST API (HTTP)
                         ▼
┌──────────────────────────────────────────────────────────┐
│               FastAPI Backend  [:8001/api]               │
│  app.py → Route handlers + Middleware + Exception hooks  │
│                                                          │
│  services/                                               │
│  ├── ingest.py       ← Encoding, normalization, schema   │
│  ├── cleaning.py     ← ETL pipeline + quality scoring    │
│  ├── eda.py          ← Profiling, plots (base64)         │
│  ├── anomaly.py      ← Isolation Forest, SVM, LOF        │
│  ├── cluster.py      ← KMeans, DBSCAN, Agglomerative     │
│  ├── advanced.py     ← PCA, t-SNE projections            │
│  ├── training.py     ← AutoML, cross-val, leaderboard    │
│  ├── predict.py      ← Single + batch inference          │
│  ├── model_registry.py ← Champion model + versioning     │
│  ├── background.py   ← Async training job queue          │
│  └── utils.py        ← Safe helpers, path management     │
│                                                          │
│  llm/                                                    │
│  ├── planner.py      ← NL → JSON DSL                     │
│  ├── validator.py    ← DSL schema validation             │
│  ├── explainer.py    ← LLM-formatted result explanations │
│  └── dsl_schema.py   ← Allowed DSL action types          │
└────────────────────────┬─────────────────────────────────┘
                         │  File I/O
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  Local Storage Layer                     │
│  data/datasets/{dataset_id}/                             │
│    ├── raw.csv           ← Original upload (untouched)   │
│    ├── clean.csv         ← ETL output                    │
│    ├── schema.json       ← Column types + metadata       │
│    ├── eda.json          ← Cached EDA results            │
│    └── meta.json         ← Training metrics + config     │
│                                                          │
│  models/{dataset_id}/                                    │
│    ├── model.pkl         ← Champion model (joblib)       │
│    ├── model_v{n}.pkl    ← Versioned history             │
│    └── registry.json     ← Model history + leaderboard   │
└──────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit 1.38 | Multi-page UI, session state, file upload |
| Backend | FastAPI 0.115 | REST API, async handling, middleware |
| Data | Pandas 2.2, NumPy 1.26 | Data manipulation, profiling |
| ML | scikit-learn 1.5 | Pipelines, models, CV, metrics |
| Boosting | XGBoost, LightGBM | Optional high-performance models |
| Imbalance | imbalanced-learn | SMOTE oversampling |
| Visualization | Matplotlib 3.9, Seaborn, Plotly | EDA plots (server-rendered, base64) |
| Encoding Detection | charset-normalizer 3.3 | Auto-detect CSV encoding |
| Date Parsing | python-dateutil 2.8 | Heuristic date format detection |
| Model Persistence | joblib 1.4 | Serialization |
| LLM Agent | Modular (OpenAI/Anthropic-compatible) | DSL planning + explanation |
| Vector Store | FAISS (local) | RAG retrieval for analytics |
| Config | pydantic-settings, python-dotenv | Environment management |
| Deployment | Docker, docker-compose | Containerized deployment |

---

## Project Structure

```
omnisearch-main/
│
├── backend/
│   ├── app.py                  # FastAPI app, all API routes
│   ├── main.py                 # App gateway, mounts /api
│   ├── config.py               # Settings via pydantic-settings
│   ├── services/
│   │   ├── ingest.py           # File parsing, encoding, schema gen
│   │   ├── cleaning.py         # ETL logic, quality scoring
│   │   ├── eda.py              # EDA functions, stat summaries
│   │   ├── anomaly.py          # Outlier detection algorithms
│   │   ├── cluster.py          # Clustering algorithms
│   │   ├── advanced.py         # PCA, t-SNE
│   │   ├── training.py         # AutoML engine (14+ models)
│   │   ├── predict.py          # Inference engine (single + batch)
│   │   ├── model_registry.py   # Champion model + version history
│   │   ├── registry.py         # Background job queue
│   │   ├── background.py       # Async training workers
│   │   └── utils.py            # Path helpers, safe serialization
│   └── llm/
│       ├── planner.py          # NL → DSL via LLM
│       ├── validator.py        # DSL schema validator
│       ├── explainer.py        # Result explanation layer
│       └── dsl_schema.py       # Allowed DSL action definitions
│
├── frontend/
│   ├── app.py                  # Streamlit entry point, session state
│   ├── theme.py                # Glassmorphism CSS injection
│   ├── utils.py                # API helpers, formatting
│   └── pages/
│       ├── Upload.py           # Dataset upload + preview
│       ├── Dashboard.py        # Executive KPI dashboard
│       ├── EDA.py              # Full EDA engine UI
│       ├── ETL.py              # Cleaning + transformation UI
│       ├── Train.py            # AutoML training UI
│       ├── Predict.py          # Inference UI (single + batch)
│       └── About.py            # Project description
│
├── data/
│   └── datasets/{dataset_id}/  # Per-dataset storage
│       ├── raw.csv
│       ├── clean.csv
│       ├── schema.json
│       ├── eda.json
│       └── meta.json
│
├── models/
│   └── {dataset_id}/           # Per-dataset model storage
│       ├── model.pkl
│       └── registry.json
│
├── tests/
│   └── test_ingest_robustness.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env
```

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- `pip` or `conda`
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/omnisearch-ai.git
cd omnisearch-ai
```

### Step 2 — Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

To enable optional boosting models:

```bash
pip install xgboost lightgbm imbalanced-learn
```

### Step 4 — Configure Environment

Create a `.env` file in the project root (or edit the existing one):

```env
DEBUG=true
DATA_DIR=data
MODELS_DIR=models
LOGS_DIR=logs

# Optional: LLM API key for NL chat features
OPENAI_API_KEY=your-key-here
# or
ANTHROPIC_API_KEY=your-key-here
```

---

## Running the Application

The system requires **two terminal windows** — one for the backend, one for the frontend.

### Terminal 1 — Start the Backend (FastAPI)

```bash
# From the project root directory
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001 --reload
```

Backend will be available at: `http://127.0.0.1:8001`  
Interactive API docs: `http://127.0.0.1:8001/api/docs`

### Terminal 2 — Start the Frontend (Streamlit)

```bash
# Wait for the backend to start, then:
cd frontend
streamlit run app.py
```

Frontend will be available at: `http://localhost:8501`

---

## Docker Deployment

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build
```

This starts both the FastAPI backend (port 8000) and Streamlit frontend (port 8501) inside a single container.



## Workflow Walkthrough

### Step 1: Upload Dataset

Navigate to the **Upload** page, select your CSV file. The system will:
- Auto-detect encoding (handles Excel-exported CSVs, UTF-16, etc.)
- Show a preview and the column normalization mapping
- Persist `raw.csv` and `schema.json` under a unique `dataset_id`

### Step 2: Explore Data (EDA)

Go to the **EDA** page and click **Run EDA**. The system generates:
- Missing value heatmap
- Distribution histograms + boxplots (before vs after outlier clipping)
- Correlation heatmap for numeric features
- Top category value counts for categorical columns
- Optional: Clustering, Anomaly Detection, PCA/t-SNE projections

### Step 3: Clean Data (ETL)

Navigate to **ETL** to configure and apply cleaning:
- Choose imputation strategy per column type
- Set outlier removal threshold (IQR multiplier or Z-score threshold)
- Preview quality score delta before committing
- System saves `clean.csv` for downstream training

### Step 4: Train Models

Go to **Train**, select a target column, and click **Train**:
- System auto-detects classification vs. regression
- Quick 10-second sample model shows immediate preview metrics
- Full training runs in background (no UI freeze)
- Results show on leaderboard: all models compared by accuracy/RMSE
- Champion model stored in `ModelRegistry`

### Step 5: Predict

Navigate to **Predict**:
- **Quick Mode** — Only the top-K most important features are shown as inputs
- **Advanced Mode** — All feature inputs visible
- Missing inputs are auto-filled from training-time defaults (median/mode)
- **Batch Upload** — Drop a CSV with multiple rows for bulk inference

---

## ML Algorithms Supported

### Classification

| Algorithm | Notes |
|-----------|-------|
| Logistic Regression | L2 regularized, `max_iter=1000` |
| Random Forest Classifier | 100 trees, default |
| Gradient Boosting Classifier | scikit-learn native |
| HistGradientBoosting Classifier | Fast, handles missing values natively |
| Decision Tree Classifier | Max depth configurable |
| Gaussian Naive Bayes | Fast probabilistic baseline |
| SGD Classifier | Online learning, log loss |
| LightGBM Classifier | Optional (if installed) |
| XGBoost Classifier | Optional (if installed) |

### Regression

| Algorithm | Notes |
|-----------|-------|
| Linear Regression | OLS |
| Ridge Regression | L2 regularized |
| Lasso Regression | L1 regularized, feature selection |
| Random Forest Regressor | 100 trees |
| Gradient Boosting Regressor | scikit-learn native |
| HistGradientBoosting Regressor | Fast native NaN support |
| Decision Tree Regressor | |
| SGD Regressor | |
| LightGBM Regressor | Optional |
| XGBoost Regressor | Optional |

### Unsupervised (EDA)

| Algorithm | Task |
|-----------|------|
| KMeans | Clustering |
| DBSCAN | Density-based clustering |
| Agglomerative | Hierarchical clustering |
| Spectral Clustering | Graph-based clustering |
| Isolation Forest | Anomaly Detection |
| One-Class SVM | Anomaly Detection |
| Local Outlier Factor | Anomaly Detection |
| PCA | Dimensionality Reduction |
| t-SNE | Visualization |

---

## Advanced Features

### Model Registry & Versioning

Every training run is logged in `registry.json` per dataset:

```json
{
  "active_version": "v3",
  "history": [
    {"version": "v1", "model": "RandomForest", "accuracy": 0.84, "timestamp": "..."},
    {"version": "v2", "model": "LightGBM",     "accuracy": 0.89, "timestamp": "..."},
    {"version": "v3", "model": "XGBoost",      "accuracy": 0.91, "timestamp": "..."}
  ]
}
```

Champion model is always auto-selected for inference.

### Pipeline Hygiene (No Data Leakage)

All preprocessing (imputation, scaling, encoding) is wrapped in a `sklearn.pipeline.Pipeline` that is **fitted only on training data** and applied to test data:

```python
pipeline = Pipeline([
    ("preprocessor", ColumnTransformer([...])),
    ("estimator", RandomForestClassifier())
])
pipeline.fit(X_train, y_train)   # fit on train only
pipeline.predict(X_test)         # apply fitted transform to test
```

### DSL-Based Safe LLM Agent

The LLM layer never executes arbitrary code. It produces structured JSON:

```json
{"action": "agg", "op": "sum", "col": "revenue"}
{"action": "groupby", "col": "region", "agg": "mean", "target": "sales"}
{"action": "plot", "kind": "hist", "col": "age"}
```

Server-side executor maps each action to a validated helper function. No `exec()`, no file system access.

### Data Quality Scoring

A 0–100 composite score computed from:
- Missing value ratio
- Mixed-type column detection
- Duplicate row ratio
- Outlier density
- Schema consistency

Shown as "Before Cleaning" vs "After Cleaning" delta.

---


## Testing

```bash
# Run all tests
pytest tests/

# Run ingestion robustness tests (malformed CSVs)
pytest tests/test_ingest_robustness.py -v

# Run training tests
python test_training_comprehensive.py

# Test anomaly detection fixes
python test_anomaly_fix.py
```



## License

MIT License. See `LICENSE` for details.

---

*Built for scale, designed for simplicity.*
