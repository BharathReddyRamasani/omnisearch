import os, json, time, joblib, hashlib, signal, random, sys, platform
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from copy import deepcopy
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold


from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, precision_recall_fscore_support
)

from backend.services.utils import load_raw, load_clean, model_dir
from backend.services.model_registry import ModelRegistry, register_trained_model


# =====================================================
# TIMEOUT HANDLER (WINDOWS-COMPATIBLE)
# =====================================================
class TimeoutError(Exception):
    pass


def set_timeout(seconds: int):
    """Set timeout - no-op on Windows"""
    import platform
    if platform.system() != "Windows":
        try:
            signal.signal(signal.SIGALRM, lambda x, y: (_ for _ in ()).throw(TimeoutError("Training timeout")))
            signal.alarm(seconds)
        except:
            pass


def cancel_timeout():
    """Cancel timeout"""
    import platform
    if platform.system() != "Windows":
        try:
            signal.alarm(0)
        except:
            pass


# =====================================================
# REPRODUCIBILITY HELPERS
# =====================================================
def set_reproducibility_seeds(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    # sklearn doesn't have global seed - use random_state parameter


def get_environment_snapshot() -> Dict:
    """Capture Python environment for reproducibility"""
    import sklearn
    return {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "platform": platform.platform()
    }


# =====================================================
# STATISTICAL DRIFT DETECTION
# =====================================================
def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI) for numeric columns.
    Measures distribution shift between expected and actual.
    PSI > 0.25 typically indicates significant drift.
    """
    try:
        # Handle missing values
        expected = expected.dropna()
        actual = actual.dropna()
        
        if len(expected) < 10 or len(actual) < 10:
            return 0.0
        
        # Create bins from expected distribution
        try:
            expected_binned = pd.cut(expected, bins=bins, duplicates='drop')
            actual_binned = pd.cut(actual, bins=expected_binned.cat.categories, duplicates='drop')
        except:
            return 0.0
        
        # Calculate proportions
        expected_prop = expected_binned.value_counts(normalize=True).sort_index()
        actual_prop = actual_binned.value_counts(normalize=True).sort_index()
        
        # Align indexes
        idx = expected_prop.index.union(actual_prop.index)
        expected_prop = expected_prop.reindex(idx, fill_value=0.0001)  # Avoid log(0)
        actual_prop = actual_prop.reindex(idx, fill_value=0.0001)
        
        # PSI = sum((actual% - expected%) * ln(actual% / expected%))
        psi = (actual_prop - expected_prop) * np.log(actual_prop / expected_prop)
        return float(psi.sum())
    except:
        return 0.0


def detect_statistical_drift(dataset_id: str, current_df: pd.DataFrame) -> Dict:
    """
    Detect data drift using statistical tests (PSI + KS-test).
    Returns drift detection with severity.
    """
    try:
        registry = ModelRegistry.load_registry()
        if dataset_id not in registry.get("models", {}):
            return {"drift_detected": False, "reason": "No previous model"}
        
        active_version = ModelRegistry.get_active_version(dataset_id)
        if not active_version:
            return {"drift_detected": False, "reason": "No active model"}
        
        # Try to load training dataset metadata
        meta = active_version.get("metadata", {})
        if "training_data_stats" not in meta:
            return {"drift_detected": False, "reason": "No baseline stats"}
        
        baseline_stats = meta["training_data_stats"]
        numeric_cols = current_df.select_dtypes(include="number").columns
        
        drift_scores = {}
        high_drift_cols = []
        
        for col in numeric_cols:
            if col not in baseline_stats:
                continue
            
            baseline = baseline_stats[col]
            current = current_df[col].dropna()
            
            if len(current) < 10:
                continue
            
            # PSI test
            psi = compute_psi(
                pd.Series(baseline.get("values", [])), 
                current, 
                bins=10
            )
            
            # KS test (Kolmogorov-Smirnov)
            baseline_vals = baseline.get("values", [])
            if baseline_vals:
                ks_stat, ks_pval = stats.ks_2samp(baseline_vals, current)
                ks_significant = ks_pval < 0.05  # 5% significance level
            else:
                ks_stat, ks_significant = 0.0, False
            
            drift_scores[col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "ks_significant": bool(ks_significant),
                "drift_detected": psi > 0.25 or ks_significant
            }
            
            if drift_scores[col]["drift_detected"]:
                high_drift_cols.append(col)
        
        drift_detected = len(high_drift_cols) > 0
        
        return {
            "drift_detected": drift_detected,
            "drift_type": "statistical",
            "columns_with_drift": high_drift_cols,
            "drift_scores": drift_scores,
            "recommendation": "Retrain model" if drift_detected else "No retraining needed"
        }
    except Exception as e:
        return {"drift_detected": False, "reason": f"Drift check skipped: {str(e)}"}


# =====================================================
# DATASET HASH & DRIFT DETECTION
# =====================================================
def dataset_hash(X: pd.DataFrame) -> str:
    """Generate MD5 hash of features (EXCLUDING TARGET to avoid unnecessary retraining)"""
    X_sorted = X[sorted(X.columns)]
    return hashlib.md5(pd.util.hash_pandas_object(X_sorted, index=True).values).hexdigest()


def detect_dataset_drift(dataset_id: str, current_X: pd.DataFrame) -> Dict:
    """Detect dataset changes (features only, excluding target)"""
    try:
        registry = ModelRegistry.load_registry()
        if dataset_id not in registry.get("models", {}):
            return {"drift_detected": False, "reason": "No previous model"}
        
        active_version = ModelRegistry.get_active_version(dataset_id)
        if not active_version:
            return {"drift_detected": False, "reason": "No active model"}
        
        stored_hash = active_version.get("metadata", {}).get("dataset_hash")
        if not stored_hash:
            return {"drift_detected": False, "reason": "No hash stored"}
        
        current_hash = dataset_hash(current_X)
        if current_hash != stored_hash:
            return {
                "drift_detected": True,
                "reason": "Dataset features changed",
                "recommendation": "Retrain model"
            }
        
        return {"drift_detected": False, "reason": "Dataset features unchanged"}
    except Exception as e:
        return {"drift_detected": False, "reason": f"Drift check skipped: {str(e)}"}


# =====================================================
# TRAINING CONFIGURATION
# =====================================================
class TrainingConfig:
    """Production-ready configuration with dataset-aware parameters"""

    @staticmethod
    def get_model_config(n_samples: int, n_features: int, task: str) -> Dict:
        """Dataset size-aware hyperparameters"""
        
        if n_samples < 1000:
            size = "small"
        elif n_samples < 10000:
            size = "medium"
        elif n_samples < 100000:
            size = "large"
        else:
            size = "xlarge"
        
        configs = {
            "classification": {
                "small": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 10, "random_state": 42},
                    "RandomForest": {"n_estimators": 50, "max_depth": 10, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 50, "max_depth": 3, "random_state": 42},
                    "NaiveBayes": {},
                },
                "medium": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 15, "random_state": 42},
                    "RandomForest": {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 100, "max_depth": 4, "random_state": 42},
                    "NaiveBayes": {},
                },
                "large": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 20, "random_state": 42},
                    "RandomForest": {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 200, "max_depth": 5, "random_state": 42},
                    "NaiveBayes": {},
                },
                "xlarge": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 25, "random_state": 42},
                    "RandomForest": {"n_estimators": 300, "max_depth": 25, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 300, "max_depth": 6, "random_state": 42},
                    "NaiveBayes": {},
                }
            },
            "regression": {
                "small": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 10, "random_state": 42},
                    "RandomForest": {"n_estimators": 50, "max_depth": 10, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 50, "max_depth": 3, "random_state": 42},
                },
                "medium": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 15, "random_state": 42},
                    "RandomForest": {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 100, "max_depth": 4, "random_state": 42},
                },
                "large": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 20, "random_state": 42},
                    "RandomForest": {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 200, "max_depth": 5, "random_state": 42},
                },
                "xlarge": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 25, "random_state": 42},
                    "RandomForest": {"n_estimators": 300, "max_depth": 25, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 300, "max_depth": 6, "random_state": 42},
                }
            }
        }
        
        return configs[task][size]

    @staticmethod
    def get_timeout_seconds(n_samples: int) -> int:
        """Adaptive timeout based on dataset size"""
        if n_samples < 1000:
            return 60
        elif n_samples < 10000:
            return 300
        elif n_samples < 50000:
            return 900
        else:
            return 1800


# =====================================================
# ID COLUMN DETECTION
# =====================================================
def load_upload_metadata(dataset_id: str) -> Dict:
    """Load upload metadata containing detected_id_columns (DATA GOVERNANCE BEST PRACTICE)"""
    try:
        from backend.services.utils import datasetdir
        metadata_path = os.path.join(datasetdir(dataset_id), "upload_metadata.json")
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_defaults(X: pd.DataFrame) -> Dict:
    """Extract medians/modes for feature defaults"""
    defaults = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            defaults[c] = float(X[c].median())
        else:
            mode_val = X[c].mode()
            defaults[c] = mode_val.iloc[0] if len(mode_val) > 0 else "UNKNOWN"
    return defaults


def extract_feature_importance(pipe, raw_columns: List, task: str) -> Dict:
    """Extract feature importance with proper preprocessing mapping"""
    model = pipe.named_steps.get("model")
    pre = pipe.named_steps["pre"]
    
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["oh"]
            try:
                feature_names.extend(ohe.get_feature_names_out(cols))
            except:
                feature_names.extend(cols)
    
    # Extract importance
    importance_method = "none"
    scores = None
    
    if hasattr(model, "feature_importances_"):
        scores = model.feature_importances_
        importance_method = "feature_importances"
    elif hasattr(model, "coef_"):
        scores = np.abs(model.coef_).ravel()
        importance_method = "absolute_coefficients"
    else:
        scores = np.ones(len(feature_names)) / len(feature_names)
        importance_method = "uniform"
    
    # Ensure length matches
    if len(scores) != len(feature_names):
        scores = np.ones(len(feature_names)) / len(feature_names)
        importance_method = "uniform"
    
    # Create dictionary and sort
    importance_dict = dict(zip(feature_names, scores))
    ranked = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    top_features = [name for name, _ in ranked[:8]]
    
    return {
        "scores": {k: round(float(v), 6) for k, v in ranked},
        "top_features": top_features,
        "method": importance_method,
        "feature_count": len(feature_names)
    }


# =====================================================
# CROSS-VALIDATION METRICS
# =====================================================
def compute_cv_metrics(pipe, X, y, task: str, cv_folds: int = 5) -> Dict:
    """Compute cross-validation metrics"""
    try:
        if task == "classification":
            scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
        else:
            scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}
        
        cv_results = cross_validate(pipe, X, y, cv=cv_folds, scoring=scoring, return_train_score=True)
        
        cv_stats = {}
        for metric in scoring.keys():
            test_scores = cv_results[f"test_{metric}"]
            train_scores = cv_results[f"train_{metric}"]
            
            cv_stats[metric] = {
                "mean": round(float(np.mean(test_scores)), 4),
                "std": round(float(np.std(test_scores)), 4),
                "fold_scores": [round(float(s), 4) for s in test_scores]
            }
        
        return {"cv_metrics": cv_stats, "cv_folds": cv_folds}
    except Exception as e:
        return {"cv_error": str(e)}


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def train_model_logic(dataset_id: str, target: str, task: str = None, test_size: float = 0.2, 
                     random_state: int = 42, time_limit_seconds: int = None) -> Dict:
    """
    PRODUCTION TRAINING WITH:
    ✅ CV-BASED MODEL SELECTION (uses CV mean score to rank models)
    ✅ Stratified train/test (classification)
    ✅ Cross-validation metrics used for model ranking
    ✅ Ridge/Lasso regularization
    ✅ Dataset-aware hyperparameters
    ✅ Adaptive timeouts
    ✅ CLEAN DATA MANDATORY (fails if not available)
    ✅ ID column filtering (detected at load time)
    ✅ Feature importance mapping
    ✅ Model versioning
    ✅ EXPLICIT task override support
    
    Args:
        task: Optional explicit override ("classification" or "regression")
              If None, auto-detect using dtype + cardinality heuristic
    """
    try:
        start_time = time.time()
        
        # ❌ FAIL if clean data doesn't exist (data leakage risk)
        # ✅ NO silent fallback to raw data
        try:
            df = load_clean(dataset_id)
            data_source = "clean"
        except Exception as clean_error:
            return {
                "status": "failed",
                "error": "Clean data not available - cannot proceed (data leakage risk)",
                "error_code": "CLEAN_DATA_REQUIRED",
                "details": f"load_clean failed: {str(clean_error)}",
                "recommendation": "Run data cleaning/ingestion first"
            }
        
        # Validate target
        if target not in df.columns:
            return {"status": "failed", "error": "Target column not found"}
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Remove missing target
        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 20:
            return {"status": "failed", "error": "Insufficient data (minimum 20 rows)"}
        
        # ✅ IMPROVED TASK DETECTION (dtype + cardinality + explicit override)
        if task is None:
            # Heuristic: check dtype first
            if pd.api.types.is_bool_dtype(y) or pd.api.types.is_integer_dtype(y):
                # Integer/bool likely classification if few unique values
                if y.nunique() <= 15:
                    task = "classification"
                else:
                    task = "regression"
            elif pd.api.types.is_float_dtype(y):
                # Float is almost always regression
                task = "regression"
            else:
                # Categorical/object: check cardinality
                if y.nunique() <= 15:
                    task = "classification"
                else:
                    task = "regression"
        
        # Validate task
        if task not in ["classification", "regression"]:
            return {"status": "failed", "error": "task must be 'classification' or 'regression'"}
        
        # Detect and remove ID columns (FROM METADATA - single source of truth)
        upload_meta = load_upload_metadata(dataset_id)
        id_columns = upload_meta.get("detected_id_columns", [])
        for col in id_columns:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # Check dataset drift (using features only)
        drift_check = detect_dataset_drift(dataset_id, X)
        
        # Set adaptive timeout
        if time_limit_seconds is None:
            time_limit_seconds = TrainingConfig.get_timeout_seconds(len(X))
        set_timeout(time_limit_seconds)
        
        # Preprocessing
        num_cols = X.select_dtypes(include="number").columns.tolist()
        cat_cols = X.select_dtypes(include="object").columns.tolist()
        
        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_cols),
        ], verbose=False)
        
        # Get model configs
        model_configs = TrainingConfig.get_model_config(len(X), len(X.columns), task)
        
        # Build models
        models = {}
        if task == "classification":
            models = {
                "LogisticRegression": LogisticRegression(**model_configs.get("LogisticRegression", {})),
                "DecisionTree": DecisionTreeClassifier(**model_configs.get("DecisionTree", {})),
                "RandomForest": RandomForestClassifier(**model_configs.get("RandomForest", {})),
                "GradientBoosting": GradientBoostingClassifier(**model_configs.get("GradientBoosting", {})),
                "NaiveBayes": GaussianNB(**model_configs.get("NaiveBayes", {})),
            }
        else:
            models = {
                "LinearRegression": LinearRegression(**model_configs.get("LinearRegression", {})),
                "Ridge": Ridge(**model_configs.get("Ridge", {})),
                "Lasso": Lasso(**model_configs.get("Lasso", {})),
                "DecisionTree": DecisionTreeRegressor(**model_configs.get("DecisionTree", {})),
                "RandomForest": RandomForestRegressor(**model_configs.get("RandomForest", {})),
                "GradientBoosting": GradientBoostingRegressor(**model_configs.get("GradientBoosting", {})),
            }
        
        # Stratified split for classification
        if task == "classification":
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y, shuffle=True
            )
        else:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                shuffle=True
            )
        
        # ✅ CAPTURE TRAINING DATA STATISTICS FOR DRIFT DETECTION
        training_data_stats = {}
        for col in X.select_dtypes(include="number").columns:
            training_data_stats[col] = {
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "values": X[col].dropna().tolist()[:1000]  # Sample for PSI
            }
        
        # Set reproducibility seeds
        set_reproducibility_seeds(random_state)
        
        # Train models with both holdout CV
        leaderboard = []
        best_pipe = None
        best_score = -1e9
        best_model_name = None
        cv_metric_used = False
        best_cv_score = None
        best_holdout_score = None
  # Track if CV was used for selection
        
        for name, model in models.items():
            try:
                model_start = time.time()
                
                # ✅ FIX 2: PREVENT PREPROCESSING LEAKAGE - CRITICAL
                # NEVER reuse fitted pipeline for CV. cross_validate clones internally.
                # Create FRESH pipeline for holdout test (fit on train split)
                holdout_pipe = Pipeline([
                    ("pre", clone(pre)),
                    ("model", model.__class__(**model.get_params()))
                ])
                holdout_pipe.fit(Xtr, ytr)
                preds = holdout_pipe.predict(Xte)
                
                # Score on holdout test set
                if task == "classification":
                    holdout_score = accuracy_score(yte, preds)
                else:
                    holdout_score = r2_score(yte, preds)
                
                # ✅ COMPUTE CV METRICS for model ranking (prevent leakage)
                cv_result = None
                cv_mean = None
                try:
                    if task == "classification":
                        from sklearn.model_selection import StratifiedKFold
                        n_splits = max(2, min(5, len(X) // 20))  # ✅ FIX: Enforce n_splits >= 2
                        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    else:
                        from sklearn.model_selection import KFold
                        n_splits = max(2, min(5, len(X) // 20))  # ✅ FIX: Enforce n_splits >= 2
                        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                    
                    # ✅ FIX 2: NEVER REUSE FITTED PIPELINE
                    # Create COMPLETELY FRESH pipeline (never fitted before CV)
                    # cross_validate will clone it per fold internally
                    cv_pipe = Pipeline([
                        ("pre", clone(pre)),
                        ("model", model.__class__(**model.get_params()))
                    ])
                    
                    # Fit on FULL X (not just train split) for CV
                    if task == "classification":
                        cv_scores = cross_validate(
                            cv_pipe, X, y, cv=cv_splitter, 
                            scoring="accuracy", n_jobs=-1, return_train_score=True
                        )
                        cv_mean = cv_scores['test_score'].mean()
                    else:
                        cv_scores = cross_validate(
                            cv_pipe, X, y, cv=cv_splitter,
                            scoring="r2", n_jobs=-1, return_train_score=True
                        )
                        cv_mean = cv_scores['test_score'].mean()
                    
                    cv_result = {
                        "cv_mean": round(float(cv_mean), 4),
                        "cv_std": round(float(cv_scores['test_score'].std()), 4),
                        "cv_fold_scores": [round(float(s), 4) for s in cv_scores['test_score']],
                        "cv_folds": len(cv_scores['test_score'])
                    }
                except Exception as cv_error:
                    cv_mean = None
                
                # Score metadata
                metrics = {}
                if task == "classification":
                    proba = None
                    try:
                        proba = holdout_pipe.predict_proba(Xte).max(axis=1).mean()  # ✅ FIX: holdout_pipe NOT pipe
                    except:
                        pass
                    
                    metrics = {
                        "accuracy": round(float(holdout_score), 4),
                        "confidence_mean": round(float(proba), 4) if proba else None
                    }
                else:
                    mae = mean_absolute_error(yte, preds)
                    rmse = np.sqrt(mean_squared_error(yte, preds))
                    
                    metrics = {
                        "r2_score": round(float(holdout_score), 4),
                        "mae": round(float(mae), 4),
                        "rmse": round(float(rmse), 4)
                    }
                
                training_time = time.time() - model_start
                
                # ✅ USE CV MEAN FOR RANKING (fallback to holdout if CV fails)
                ranking_score = cv_mean if cv_mean is not None else holdout_score
                
                entry = {
                    "model": name,
                    "holdout_score": round(float(holdout_score), 4),
                    "ranking_score": round(float(ranking_score), 4),
                    "cv_metrics": cv_result,
                    "metrics": metrics,
                    "train_rows": int(len(Xtr)),
                    "test_rows": int(len(Xte)),
                    "training_time_seconds": round(training_time, 2)
                }
                
                leaderboard.append(entry)
                
                # ✅ SELECT BEST BY CV MEAN (or holdout if no CV)
                if ranking_score > best_score:
                    best_score = ranking_score
                    best_pipe = holdout_pipe
                    best_model_name = name
                    cv_metric_used = (cv_mean is not None)
                    # ✅ Track both scores explicitly
                    best_cv_score = cv_mean
                    best_holdout_score = holdout_score

            
            except Exception as e:
                leaderboard.append({
                    "model": name,
                    "score": None,
                    "error": str(e)
                })
        
        cancel_timeout()
        
        if best_pipe is None:
            return {"status": "failed", "error": "All models failed to train"}
        
        # ✅ FIX #4: HYPERPARAMETER TUNING WITH RandomizedSearchCV
        # Even 5-10 iterations gives us real tuning (not just rule-based)
        hyperparameter_search_info = {
            "method": "RandomizedSearchCV",
            "iterations": 5,
            "models_tuned": []
        }
        
        try:
            # Only tune ensemble methods (they benefit most from tuning)
            tunable_models = {}
            if task == "classification":
                if best_model_name == "RandomForest":
                    tunable_models["RandomForest"] = {
                        "model": RandomForestClassifier(random_state=random_state),
                        "params": {
                            "n_estimators": [50, 100, 150],
                            "max_depth": [5, 10, 15, None],
                            "min_samples_split": [2, 5],
                            "min_samples_leaf": [1, 2]
                        }
                    }
                elif best_model_name == "GradientBoosting":
                    tunable_models["GradientBoosting"] = {
                        "model": GradientBoostingClassifier(random_state=random_state),
                        "params": {
                            "n_estimators": [50, 100, 150],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "min_samples_split": [2, 5]
                        }
                    }
            else:
                if best_model_name == "RandomForest":
                    tunable_models["RandomForest"] = {
                        "model": RandomForestRegressor(random_state=random_state),
                        "params": {
                            "n_estimators": [50, 100, 150],
                            "max_depth": [5, 10, 15, None],
                            "min_samples_split": [2, 5],
                            "min_samples_leaf": [1, 2]
                        }
                    }
                elif best_model_name == "GradientBoosting":
                    tunable_models["GradientBoosting"] = {
                        "model": GradientBoostingRegressor(random_state=random_state),
                        "params": {
                            "n_estimators": [50, 100, 150],
                            "learning_rate": [0.01, 0.05, 0.1],
                            "max_depth": [3, 5, 7],
                            "min_samples_split": [2, 5]
                        }
                    }
            
            # Run RandomizedSearchCV only for tunable models
            best_tuned_model = None
            best_tuned_score = best_score
            best_tuned_params = None
            
            for tune_name, tune_config in tunable_models.items():
                try:
                    if task == "classification":
                        scorer = "accuracy"

                        cv_splitter = StratifiedKFold(
                            n_splits=min(3, max(2, len(Xtr) // 50)),
                            shuffle=True,
                            random_state=random_state
                        )
                    else:
                        scorer = "r2"
                        cv_splitter = KFold(
                            n_splits=min(3, max(2, len(Xtr) // 50)),
                            shuffle=True,
                            random_state=random_state
                        )

                    # ✅ INDUSTRIAL FIX: Wrap model in pipeline for preprocessing-aware tuning
                    # This ensures tuned model behaves identically to production pipeline
                    tune_pipe = Pipeline([
                        ("pre", clone(pre)),  # Include preprocessing
                        ("model", tune_config["model"])
                    ])
                    
                    # Build parameter grid with pipeline prefix
                    param_grid = {}
                    for param_name, param_values in tune_config["params"].items():
                        param_grid[f"model__{param_name}"] = param_values
                    
                    # RandomizedSearchCV: 5 iterations only (fast)
                    
                    search = RandomizedSearchCV(
                        tune_pipe,
                        param_grid,
                        n_iter=min(5, len(Xtr) // 100),
                        cv=cv_splitter,
                        scoring=scorer,
                        n_jobs=-1,
                        random_state=random_state,
                        verbose=0
                    )
                    
                    # Fit on holdout train split (preprocessing happens inside pipeline)
                    search.fit(Xtr, ytr)
                    tuned_score = search.score(Xte, yte)
                    
                    # If tuned model beats best, use it
                    if tuned_score > best_tuned_score:
                        best_tuned_score = tuned_score
                        best_tuned_model = search.best_estimator_  # Entire pipeline
                        best_tuned_params = search.best_params_
                        best_score = tuned_score
                    
                    hyperparameter_search_info["models_tuned"].append({
                        "model": tune_name,
                        "best_params": best_tuned_params,
                        "best_cv_score": round(float(search.best_score_), 4),
                        "test_score": round(float(tuned_score), 4),
                        "pipeline_aware": True  # ✅ Mark as pipeline-aware
                    })
                except:
                    pass  # If tuning fails, continue with rule-based
        except:
            pass  # If tuning disabled, continue with rule-based
        
        # Use tuned model if found, else use best rule-based model
        if best_tuned_model is not None:
            # best_tuned_model is already a complete pipeline
            production_pipe = best_tuned_model
            hyperparameter_search_info["used_tuned_model"] = True
        else:
            # Use best rule-based pipeline as-is
            production_pipe = best_pipe
            hyperparameter_search_info["used_tuned_model"] = False
        
        # ✅ Refit production pipeline on FULL dataset (X, y) - not just training split
        # Pipeline already has preprocessing + model, just re-fit with new data
        production_pipe.fit(X, y)
        
        # Verify production model works
        try:
            _ = production_pipe.predict(X[:5])
        except:
            return {"status": "failed", "error": "Production model failed verification"}
        
        # Extract feature importance from PRODUCTION model
        importance_data = extract_feature_importance(production_pipe, list(X.columns), task)
        
        # Cross-validation metrics on production model
        cv_data = compute_cv_metrics(production_pipe, X, y, task, cv_folds=min(5, len(X) // 10))
        
        # Save PRODUCTION model (trained on full dataset)
        root = model_dir(dataset_id)
        joblib.dump(production_pipe, os.path.join(root, "model.pkl"))
        
        total_time = time.time() - start_time
        
        # Get environment snapshot for reproducibility
        env_snapshot = get_environment_snapshot()
        
        # Use statistical drift instead of hash-only
        statistical_drift = detect_statistical_drift(dataset_id, df)
        
        # ✅ FIX #3: ENFORCE DRIFT DETECTION
        # When drift detected, downgrade confidence and flag for retraining
        drift_confidence_adjustment = 1.0  # Multiplier for prediction confidence
        drift_requires_retraining = False
        
        if statistical_drift.get("drift_detected", False):
            drift_confidence_adjustment = 0.85  # Reduce confidence by 15%
            drift_requires_retraining = True  # Flag for retraining recommendation
        
        result = {
            "status": "ok",
            "schema_version": "2.1",  # Incremented: now includes training_data_stats
            "dataset_id": dataset_id,
            "task": task,
            "target": target,
            "best_model": best_model_name,
            "selection_score": round(float(best_score), 4),
            "selection_basis": "cv_mean" if cv_metric_used else "holdout",
            "cv_mean_score": round(float(best_cv_score), 4) if best_cv_score is not None else None,
            "holdout_score": round(float(best_holdout_score), 4),
            "training_approach": (
                "Model selected using cross-validation mean score "
                "with holdout fallback; final model retrained on full dataset"
            ),
            "leaderboard": leaderboard,
            "raw_columns": list(X.columns),
            "dropped_id_columns": id_columns,
            "feature_defaults": extract_defaults(X),
            "feature_importance": importance_data,
            "top_features": importance_data["top_features"],
            "cross_validation": cv_data,
            "data_source": data_source,
            "training_time_seconds": round(total_time, 2),
            "timeout_limit_seconds": time_limit_seconds,
            "dataset_size_category": "small" if len(X) < 1000 else "medium" if len(X) < 10000 else "large",
            
            # ✅ FIX #4: ACTUAL HYPERPARAMETER TUNING
            "hyperparameter_tuning": hyperparameter_search_info,
            
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # ✅ REPRODUCIBILITY
            "environment": env_snapshot,
            "random_state": random_state,
            
            # ✅ TRAINING DATA STATISTICS (for drift detection)
            "training_data_stats": training_data_stats,
            
            # ✅ FIX #3: STATISTICAL DRIFT WITH ENFORCEMENT
            "statistical_drift": statistical_drift,
            "drift_enforced": True,
            "drift_confidence_adjustment": drift_confidence_adjustment,
            "drift_requires_retraining": drift_requires_retraining,
            
            # ✅ FIX #4: GOVERNANCE CLARIFICATION
            "version_governance": {
                "authority": "ModelRegistry",
                "training_role": "Trains model and emits metadata only",
                "registry_role": "Single source of truth for versioning",
                "version_assignment": "Automatic (registry-controlled)"
        },
            
            # ✅ DATASET HASH (for completeness - features only)
            "dataset_hash": dataset_hash(X),
        }
        
        if statistical_drift["drift_detected"]:
            result["statistical_drift_warning"] = statistical_drift
            result["dataset_drift_warning"] = drift_check
        
        # Save metadata
        with open(os.path.join(root, "metadata.json"), "w") as f:
            json.dump(result, f, indent=2)
        
        # Register in model registry
        try:
            registry_result = register_trained_model(dataset_id, result)
            result["registry_info"] = registry_result
        except:
            pass
        
        return result
    
    except TimeoutError:
        return {"status": "failed", "error": "Training exceeded time limit"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
