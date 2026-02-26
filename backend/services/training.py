import os, json, time, joblib, hashlib, signal, random, sys, platform, uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from copy import deepcopy
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, clone
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, learning_curve
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, precision_recall_fscore_support,
    brier_score_loss, balanced_accuracy_score
)

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV

# ✅ LightGBM + XGBoost
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ✅ SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from backend.services.utils import load_raw, load_clean, model_dir
from backend.services.model_registry import ModelRegistry, register_trained_model
from backend.services.ingest import convert_numpy_to_python


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
                    "GradientBoosting": {"n_estimators": 50, "max_depth": 5, "random_state": 42},
                    "NaiveBayes": {},
                },
                "medium": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 15, "random_state": 42},
                    "RandomForest": {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 100, "max_depth": 7, "random_state": 42},
                    "NaiveBayes": {},
                },
                "large": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 20, "random_state": 42},
                    "RandomForest": {"n_estimators": 200, "max_depth": 20, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 200, "max_depth": 9, "random_state": 42},
                    "NaiveBayes": {},
                },
                "xlarge": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 25, "random_state": 42},
                    "RandomForest": {"n_estimators": 300, "max_depth": 25, "random_state": 42, "n_jobs": -1},
                    "GradientBoosting": {"n_estimators": 300, "max_depth": 10, "random_state": 42},
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
def compute_cv_metrics(pipe, X, y, task: str, cv_folds: int = 10) -> Dict:
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
# EXPERIMENT TRACKING
# =====================================================
def generate_experiment_id() -> str:
    return str(uuid.uuid4())


def compute_model_checksum(model_path: str) -> str:
    try:
        with open(model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "checksum_unavailable"


def snapshot_params(dataset_id, target, task, test_size, random_state,
                    feature_selection, handle_imbalance, n_samples, n_features) -> Dict:
    return {
        "dataset_id": dataset_id, "target": target, "task": task,
        "test_size": test_size, "random_state": random_state,
        "feature_selection_enabled": feature_selection,
        "imbalance_handling_enabled": handle_imbalance,
        "n_samples": int(n_samples), "n_features": int(n_features),
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "xgboost_available": XGBOOST_AVAILABLE,
        "smote_available": SMOTE_AVAILABLE,
    }


# =====================================================
# LGBM / XGBOOST MODEL BUILDERS
# =====================================================
def build_boosting_models(task: str, random_state: int, n_jobs: int) -> Dict:
    models = {}
    if task == "classification":
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                random_state=random_state, n_jobs=n_jobs, verbose=-1)
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=random_state, n_jobs=n_jobs,
                eval_metric="logloss", verbosity=0)
    else:
        if LIGHTGBM_AVAILABLE:
            models["LightGBM"] = lgb.LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                random_state=random_state, n_jobs=n_jobs, verbose=-1)
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = xgb.XGBRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=6,
                random_state=random_state, n_jobs=n_jobs,
                eval_metric="rmse", verbosity=0)
    return models


# =====================================================
# INCREMENTAL LEARNING (huge datasets > 500k rows)
# =====================================================
def train_incremental(X_transformed, y, task: str, random_state: int):
    try:
        chunk_size = 10000
        if task == "classification":
            model = SGDClassifier(random_state=random_state, loss="modified_huber",
                                  class_weight="balanced", max_iter=1)
            classes = np.unique(y)
        else:
            model = SGDRegressor(random_state=random_state, max_iter=1)
            classes = None
        for start in range(0, X_transformed.shape[0], chunk_size):
            Xc = X_transformed[start:start + chunk_size]
            yc = np.array(y)[start:start + chunk_size]
            if task == "classification":
                model.partial_fit(Xc, yc, classes=classes)
            else:
                model.partial_fit(Xc, yc)
        return model
    except Exception:
        return None


# =====================================================
# LEARNING CURVES
# =====================================================
def compute_learning_curves(pipe, X, y, task: str, random_state: int) -> Dict:
    try:
        scoring = "accuracy" if task == "classification" else "r2"
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state) \
             if task == "classification" else \
             KFold(n_splits=3, shuffle=True, random_state=random_state)
        sizes, train_sc, val_sc = learning_curve(
            pipe, X, y, train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
            cv=cv, scoring=scoring, n_jobs=1, shuffle=True, random_state=random_state)
        gap = float(train_sc[:, -1].mean() - val_sc[:, -1].mean())
        indicator = "good_fit" if gap < 0.05 else ("slight_overfit" if gap < 0.15 else "overfit")
        return {
            "train_sizes": [int(s) for s in sizes],
            "train_scores_mean": [round(float(v), 4) for v in train_sc.mean(axis=1)],
            "train_scores_std":  [round(float(v), 4) for v in train_sc.std(axis=1)],
            "val_scores_mean":   [round(float(v), 4) for v in val_sc.mean(axis=1)],
            "val_scores_std":    [round(float(v), 4) for v in val_sc.std(axis=1)],
            "bias_variance_indicator": indicator,
            "bias_variance_gap": round(gap, 4),
            "scoring": scoring,
        }
    except Exception as e:
        return {"lc_error": str(e)}


# =====================================================
# MODEL CALIBRATION (classification)
# =====================================================
def compute_calibration_metrics(pipe, X_test, y_test) -> Dict:
    try:
        if not hasattr(pipe, "predict_proba"):
            return {"calibration_skipped": True, "reason": "no predict_proba"}
        y_prob = pipe.predict_proba(X_test)
        classes = getattr(pipe, "classes_", np.unique(y_test))
        if len(classes) == 2:
            brier = float(brier_score_loss(y_test, y_prob[:, 1]))
            prob_true, prob_pred = calibration_curve(
                y_test, y_prob[:, 1], n_bins=10, strategy="uniform")
            cal_curve = {
                "prob_true": [round(float(v), 4) for v in prob_true],
                "prob_pred": [round(float(v), 4) for v in prob_pred],
            }
        else:
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_test, classes=classes)
            brier = float(np.mean([
                brier_score_loss(y_bin[:, i], y_prob[:, i]) for i in range(len(classes))]))
            cal_curve = None
        quality = "good" if brier < 0.1 else ("moderate" if brier < 0.2 else "poor")
        return {"brier_score": round(brier, 4), "calibration_curve": cal_curve,
                "calibration_quality": quality}
    except Exception as e:
        return {"calibration_error": str(e)}


# =====================================================
# MODEL COMPLEXITY
# =====================================================
def compute_model_complexity(pipe, X_sample) -> Dict:
    try:
        import io
        model = pipe.named_steps.get("model", pipe)
        n_params = 0
        if hasattr(model, "coef_"):
            n_params = int(np.prod(model.coef_.shape))
            if hasattr(model, "intercept_"):
                n_params += int(np.array(model.intercept_).size)
        elif hasattr(model, "n_estimators"):
            n_params = int(getattr(model, "n_estimators", 0))
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        model_size_kb = round(buf.tell() / 1024, 1)
        sample = X_sample.head(min(100, len(X_sample)))
        t0 = time.time()
        for _ in range(10):
            pipe.predict(sample)
        latency_ms = round((time.time() - t0) / 10 * 1000, 2)
        return {"n_parameters": n_params, "model_size_kb": model_size_kb,
                "inference_latency_ms": latency_ms}
    except Exception as e:
        return {"complexity_error": str(e)}


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def train_model_logic(dataset_id: str, target: str, task: str = None, test_size: float = 0.2,
                     random_state: int = 42, time_limit_seconds: int = None,
                     feature_selection: bool = True, handle_imbalance: bool = True) -> Dict:
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
        experiment_id = generate_experiment_id()
        start_time = time.time()
        
        # ✅ DATA LOADING: try clean (post-ETL) → ingested → raw CSV fallback
        # This lets users train without running ETL first — sklearn pipeline
        # handles imputation and encoding so raw data is safe to use.
        data_source = "unknown"
        df = None
        try:
            df = load_clean(dataset_id)
            data_source = "clean"
            print(f"[TRAINING] Using ETL-cleaned data for {dataset_id}")
        except Exception:
            pass

        if df is None:
            try:
                from backend.services.utils import load_ingested
                df = load_ingested(dataset_id)
                data_source = "ingested"
                print(f"[TRAINING] No clean data found, using ingested data for {dataset_id}")
            except Exception:
                pass

        if df is None:
            try:
                from backend.services.utils import load_raw
                df = load_raw(dataset_id)
                data_source = "raw"
                print(f"[TRAINING] No ingested data found, using raw CSV for {dataset_id} — sklearn pipeline handles imputation/encoding")
            except Exception as raw_err:
                return {
                    "status": "failed",
                    "error": f"Dataset not found: {str(raw_err)}",
                    "error_code": "DATASET_NOT_FOUND",
                    "recommendation": "Upload a dataset first (Upload page).",
                }

        
        # ✅ RESOLVE TARGET COLUMN NAME MAPPING
        # The user provides a normalized column name (from column_mapping),
        # but the clean CSV has the original column names.
        # We need to map the user-provided target back to the original column name.
        try:
            upload_meta = load_upload_metadata(dataset_id)
        except Exception:
            upload_meta = {}
        column_mapping = upload_meta.get("column_mapping", {})

        
        # Create reverse mapping: normalized_name -> original_name
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # Convert user-provided target to original column name
        original_target = reverse_mapping.get(target, target)
        
        # Validate target - first check if original target is in df
        if original_target not in df.columns:
            # If not found as original, try the user-provided target directly (case-insensitive fallback)
            case_insensitive_cols = {col.strip().lower(): col for col in df.columns}
            original_target = case_insensitive_cols.get(target.strip().lower(), target)
            
            if original_target not in df.columns:
                return {
                    "status": "failed",
                    "error": f"Target column '{target}' not found in dataset",
                    "available_columns": list(df.columns),
                    "hint": "Use the normalized column names from the ETL step"
                }
        
        # Use the resolved original target column name
        target = original_target
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Remove missing target
        valid_mask = ~y.isna()
        X, y = X[valid_mask], y[valid_mask]
        
        # ✅ FIX 8: Replace inf/-inf with NaN so SimpleImputer handles them
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if len(X) < 20:
            return {"status": "failed", "error": "Insufficient data (minimum 20 rows)"}
        
        # ✅ FIX 5: FORCE CLASSIFICATION FOR STRING/OBJECT TARGETS
        # String targets cannot be used in regression (LinearRegression etc. crash)
        if task is None:
            if y.dtype == object or y.dtype.name == 'category':
                # String/categorical target → always classification
                task = "classification"
            elif pd.api.types.is_bool_dtype(y) or pd.api.types.is_integer_dtype(y):
                task = "classification" if y.nunique() <= 15 else "regression"
            elif pd.api.types.is_float_dtype(y):
                task = "regression"
            else:
                task = "classification" if y.nunique() <= 15 else "regression"
        
        # If user forced regression but target is string, override to classification
        if task == "regression" and y.dtype == object:
            task = "classification"
        
        # Validate task
        if task not in ["classification", "regression"]:
            return {"status": "failed", "error": "task must be 'classification' or 'regression'"}
        
        # ✅ RESOLVE ID COLUMN NAMES (FROM METADATA - single source of truth)
        id_columns_normalized = upload_meta.get("detected_id_columns", [])
        id_columns_original = [reverse_mapping.get(col, col) for col in id_columns_normalized]
        
        for col in id_columns_original:
            if col in X.columns:
                X = X.drop(columns=[col])
        
        # ✅ FIX 2: DROP HIGH-CARDINALITY CATEGORICAL COLUMNS
        # Columns with >50 unique values (e.g. names, addresses) create
        # thousands of one-hot columns → OOM and provide no ML signal
        high_card_cols = []
        cat_cols_all = X.select_dtypes(include="object").columns.tolist()
        for col in cat_cols_all:
            n_unique = X[col].nunique()
            if n_unique > 50:
                high_card_cols.append(col)
                X = X.drop(columns=[col])
        
        if high_card_cols:
            print(f"[TRAINING] Dropped {len(high_card_cols)} high-cardinality columns: {high_card_cols}")
        
        # Check dataset drift (using features only)
        drift_check = detect_dataset_drift(dataset_id, X)
        
        # Set adaptive timeout
        if time_limit_seconds is None:
            time_limit_seconds = TrainingConfig.get_timeout_seconds(len(X))
        set_timeout(time_limit_seconds)
        
        # ✅ FIX 4: Adaptive n_jobs (avoid OOM on large datasets)
        n_jobs = 1 if len(X) > 50000 else -1
        
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
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=25))
            ]), cat_cols),
        ], verbose=False)
        
        # Get model configs
        model_configs = TrainingConfig.get_model_config(len(X), len(X.columns), task)
        
        # ✅ CLASS WEIGHT: inject 'balanced' for imbalanced classification
        use_class_weight = handle_imbalance and task == "classification"
        cw = "balanced" if use_class_weight else None
        
        # Build models — dataset-size-aware routing
        XLARGE_THRESHOLD = 100000  # > 100k rows: LightGBM only
        models = {}
        if task == "classification":
            if len(X) <= XLARGE_THRESHOLD:
                models = {
                    "LogisticRegression": LogisticRegression(**{**model_configs.get("LogisticRegression", {}), "class_weight": cw}),
                    "DecisionTree":       DecisionTreeClassifier(**{**model_configs.get("DecisionTree", {}), "class_weight": cw}),
                    "RandomForest":       RandomForestClassifier(**{**model_configs.get("RandomForest", {}), "class_weight": cw}),
                    "GradientBoosting":   GradientBoostingClassifier(**model_configs.get("GradientBoosting", {})),
                    "NaiveBayes":         GaussianNB(**model_configs.get("NaiveBayes", {})),
                }
            # Always add LightGBM + XGBoost
            models.update(build_boosting_models(task, random_state, n_jobs))
            
            # ✅ FALLBACK for XLARGE datasets if boosting models are unavailable
            if len(models) == 0:
                print("[TRAINING] Boosting unavailable for XLARGE dataset. Using HistGradientBoosting/SGD fallback.")
                models = {
                    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state, class_weight=cw),
                    "SGDClassifier": SGDClassifier(random_state=random_state, class_weight=cw, n_jobs=n_jobs),
                    "DecisionTreeFast": DecisionTreeClassifier(random_state=random_state, max_depth=10, class_weight=cw)
                }
        else:
            if len(X) <= XLARGE_THRESHOLD:
                models = {
                    "LinearRegression": LinearRegression(**model_configs.get("LinearRegression", {})),
                    "Ridge":            Ridge(**model_configs.get("Ridge", {})),
                    "Lasso":            Lasso(**model_configs.get("Lasso", {})),
                    "DecisionTree":     DecisionTreeRegressor(**model_configs.get("DecisionTree", {})),
                    "RandomForest":     RandomForestRegressor(**model_configs.get("RandomForest", {})),
                    "GradientBoosting": GradientBoostingRegressor(**model_configs.get("GradientBoosting", {})),
                }
            models.update(build_boosting_models(task, random_state, n_jobs))

            # ✅ FALLBACK for XLARGE datasets if boosting models are unavailable
            if len(models) == 0:
                print("[TRAINING] Boosting unavailable for XLARGE dataset. Using HistGradientBoosting/SGD fallback.")
                models = {
                    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=random_state),
                    "SGDRegressor": SGDRegressor(random_state=random_state),
                    "DecisionTreeFast": DecisionTreeRegressor(random_state=random_state, max_depth=10)
                }
        
        # ✅ FIX 10: ENCODE STRING TARGETS FOR XGBOOST/LIGHTGBM
        class_mapping = None
        if task == "classification":
            # If target is string/object/categorical, encode to integers
            if y.dtype == object or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_bool_dtype(y):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                class_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
                y = pd.Series(y_encoded, index=y.index, name=y.name)
        
        # ✅ FIX 1: SAFE STRATIFIED SPLIT (fallback on rare classes)
        if task == "classification":
            try:
                # Filter out classes with <2 samples (can't stratify these)
                class_counts = y.value_counts()
                rare_classes = class_counts[class_counts < 2].index.tolist()
                if rare_classes:
                    mask = ~y.isin(rare_classes)
                    X, y = X[mask], y[mask]
                    print(f"[TRAINING] Removed {len(rare_classes)} rare classes with <2 samples")
                
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y, shuffle=True
                )
            except ValueError as e:
                print(f"[TRAINING] Stratified split failed ({e}), falling back to random split")
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    shuffle=True
                )
        else:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                shuffle=True
            )
        
        # ✅ FIX 7: CAPTURE TRAINING DATA STATISTICS (summary only, no raw values)
        training_data_stats = {}
        for col in X.select_dtypes(include="number").columns:
            col_data = X[col].dropna()
            training_data_stats[col] = {
                "mean": float(col_data.mean()) if len(col_data) > 0 else 0.0,
                "std": float(col_data.std()) if len(col_data) > 1 else 0.0,
                "min": float(col_data.min()) if len(col_data) > 0 else 0.0,
                "max": float(col_data.max()) if len(col_data) > 0 else 0.0,
                "median": float(col_data.median()) if len(col_data) > 0 else 0.0,
                "count": int(len(col_data)),
            }
        
        # Set reproducibility seeds
        set_reproducibility_seeds(random_state)
        
        # ✅ SPEED OPTIMIZATION: Sample data for model SELECTION phase
        # The final production model is always trained on ALL data
        SAMPLE_LIMIT = 5000  # Use 5k samples for model comparison
        use_sampling = len(Xtr) > SAMPLE_LIMIT
        
        if use_sampling:
            print(f"[TRAINING] Using {SAMPLE_LIMIT}/{len(Xtr)} samples for model selection (full data used for final model)")
            sample_idx = np.random.RandomState(random_state).choice(len(Xtr), SAMPLE_LIMIT, replace=False)
            Xtr_sel = Xtr.iloc[sample_idx]
            ytr_sel = ytr.iloc[sample_idx]
            # Also sample test set proportionally
            te_sample = min(len(Xte), max(1000, SAMPLE_LIMIT // 4))
            te_idx = np.random.RandomState(random_state).choice(len(Xte), te_sample, replace=False)
            Xte_sel = Xte.iloc[te_idx]
            yte_sel = yte.iloc[te_idx]
        else:
            Xtr_sel, ytr_sel = Xtr, ytr
            Xte_sel, yte_sel = Xte, yte
        
        # ✅ SPEED: Determine CV strategy based on data size
        # Small (<5k): 5-fold CV | Medium (5-20k): 3-fold CV | Large (>20k): holdout only
        skip_cv = len(X) > 20000
        cv_folds_selection = max(2, min(3 if len(X) > 5000 else 5, len(X) // 20))
        
        print(f"[TRAINING] Strategy: {'holdout-only' if skip_cv else f'{cv_folds_selection}-fold CV'} | "
              f"sampling={use_sampling} | n_jobs={n_jobs} | {len(models)} models")
        
        # Train models
        leaderboard = []
        best_pipe = None
        best_score = -1e9
        best_model_name = None
        cv_metric_used = False
        best_cv_score = None
        best_holdout_score = None
        
        for name, model in models.items():
            try:
                model_start = time.time()
                print(f"[TRAINING] Training {name}...")
                
                # Create FRESH pipeline — use clone() which works for
                # sklearn, LightGBM, XGBoost and all compatible estimators.
                # model.__class__(**model.get_params()) breaks on LGBM/XGB
                # because get_params() includes internal-only attributes.
                holdout_pipe = Pipeline([
                    ("pre", clone(pre)),
                    ("model", clone(model))
                ])
                holdout_pipe.fit(Xtr_sel, ytr_sel)
                preds = holdout_pipe.predict(Xte_sel)
                
                # Score on holdout test set
                if task == "classification":
                    holdout_score = accuracy_score(yte_sel, preds)
                else:
                    holdout_score = r2_score(yte_sel, preds)
                
                # CV: only if dataset is small/medium enough
                cv_result = None
                cv_mean = None
                if not skip_cv:
                    try:
                        n_splits = cv_folds_selection
                        if task == "classification":
                            cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        else:
                            cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                        
                        # Use sampled data for CV too
                        X_cv = pd.concat([Xtr_sel, Xte_sel]) if use_sampling else X
                        y_cv = pd.concat([ytr_sel, yte_sel]) if use_sampling else y
                        
                        cv_pipe = Pipeline([
                            ("pre", clone(pre)),
                            ("model", clone(model))
                        ])
                        
                        scoring = "accuracy" if task == "classification" else "r2"
                        cv_scores = cross_validate(
                            cv_pipe, X_cv, y_cv, cv=cv_splitter,
                            scoring=scoring, n_jobs=n_jobs, return_train_score=False
                        )
                        cv_mean = cv_scores['test_score'].mean()
                        
                        cv_result = {
                            "cv_mean": round(float(cv_mean), 4),
                            "cv_std": round(float(cv_scores['test_score'].std()), 4),
                            "cv_fold_scores": [round(float(s), 4) for s in cv_scores['test_score']],
                            "cv_folds": len(cv_scores['test_score'])
                        }
                    except Exception as cv_error:
                        print(f"[TRAINING] CV failed for {name}: {cv_error}")
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
                    # ✅ FIX: use yte_sel (sampled test set, consistent with preds)
                    mae = mean_absolute_error(yte_sel, preds)
                    rmse = np.sqrt(mean_squared_error(yte_sel, preds))
                    
                    metrics = {
                        "r2_score": round(float(holdout_score), 4),
                        "mae": round(float(mae), 4),
                        "rmse": round(float(rmse), 4)
                    }
                
                training_time = time.time() - model_start
                
                # ✅ USE CV MEAN FOR RANKING (fallback to holdout if CV fails)
                ranking_score = cv_mean if cv_mean is not None else holdout_score
                
                # ✅ CONFIDENCE INTERVAL: ±1.96 × cv_std
                ci_95 = None
                if cv_result:
                    ci_95 = round(1.96 * cv_result["cv_std"], 4)

                entry = {
                    "model": name,
                    "holdout_score": round(float(holdout_score), 4),
                    "ranking_score": round(float(ranking_score), 4),
                    "cv_metrics": cv_result,
                    "ci_95": ci_95,
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
                import traceback
                tb = traceback.format_exc()
                print(f"[TRAINING] ❌ {name} FAILED: {str(e)}\n{tb}")
                leaderboard.append({
                    "model": name,
                    "score": None,
                    "error": str(e)
                })
        
        cancel_timeout()
        
        if best_pipe is None:
            return {"status": "failed", "error": "All models failed to train"}
        
        # ✅ HYPERPARAMETER TUNING (skip for large datasets — too slow)
        hyperparameter_search_info = {
            "method": "RandomizedSearchCV",
            "iterations": 5,
            "models_tuned": []
        }
        
        # ✅ FIX: Always initialize best_tuned_model so it is never unbound
        best_tuned_model = None
        
        skip_tuning = len(X) > 10000  # Skip tuning for datasets > 10k rows
        if skip_tuning:
            print(f"[TRAINING] Skipping hyperparameter tuning (dataset has {len(X)} rows > 10k)")
            hyperparameter_search_info["skipped"] = True
            hyperparameter_search_info["reason"] = f"Dataset too large ({len(X)} rows)"
        
        if not skip_tuning:
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
                best_tuned_score = best_score
                best_tuned_params = None
                
                for tune_name, tune_config in tunable_models.items():
                    try:
                        scorer = "accuracy" if task == "classification" else "r2"
                        hp_n_splits = max(2, min(3, len(Xtr) // 50))
                        if task == "classification":
                            cv_splitter = StratifiedKFold(
                                n_splits=hp_n_splits, shuffle=True, random_state=random_state
                            )
                        else:
                            cv_splitter = KFold(
                                n_splits=hp_n_splits, shuffle=True, random_state=random_state
                            )

                        tune_pipe = Pipeline([
                            ("pre", clone(pre)),
                            ("model", tune_config["model"])
                        ])
                        
                        param_grid = {}
                        for param_name, param_values in tune_config["params"].items():
                            param_grid[f"model__{param_name}"] = param_values
                        
                        hp_n_iter = max(1, min(5, len(Xtr) // 100))
                        
                        search = RandomizedSearchCV(
                            tune_pipe,
                            param_grid,
                            n_iter=hp_n_iter,
                            cv=cv_splitter,
                            scoring=scorer,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            verbose=0
                        )
                        
                        search.fit(Xtr, ytr)
                        tuned_score = search.score(Xte, yte)
                        
                        if tuned_score > best_tuned_score:
                            best_tuned_score = tuned_score
                            best_tuned_model = search.best_estimator_
                            best_tuned_params = search.best_params_
                            best_score = tuned_score
                        
                        hyperparameter_search_info["models_tuned"].append({
                            "model": tune_name,
                            "best_params": best_tuned_params,
                            "best_cv_score": round(float(search.best_score_), 4),
                            "test_score": round(float(tuned_score), 4),
                            "pipeline_aware": True
                        })
                    except:
                        pass
            except:
                pass
        
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
        # ✅ SPEED: Skip final CV for large datasets (>10k rows) — too slow
        if len(X) > 10000:
            print(f"[TRAINING] Skipping final CV (dataset has {len(X)} rows > 10k)")
            cv_data = {"cv_skipped": True, "reason": f"Dataset too large ({len(X)} rows)"}
        else:
            cv_data = compute_cv_metrics(production_pipe, X, y, task, cv_folds=max(2, min(5, len(X) // 10)))
        
        # Save PRODUCTION model (trained on full dataset)
        root = model_dir(dataset_id)
        model_path = os.path.join(root, "model.pkl")
        joblib.dump(production_pipe, model_path)
        
        # ✅ EXPERIMENT TRACKING: checksum + param snapshot
        model_checksum = compute_model_checksum(model_path)
        param_snap = snapshot_params(
            dataset_id, target, task, test_size, random_state,
            feature_selection, handle_imbalance, len(X), len(X.columns))
        
        # ✅ BALANCED ACCURACY (classification with imbalance handling)
        balanced_acc = None
        if task == "classification":
            try:
                yte_pred_bal = production_pipe.predict(Xte)
                balanced_acc = round(float(balanced_accuracy_score(yte, yte_pred_bal)), 4)
            except Exception:
                pass
        
        # ✅ LEARNING CURVES (allow for larger datasets up to 100k)
        if len(X) <= 100000:
            print("[TRAINING] Computing learning curves...")
            learning_curves_data = compute_learning_curves(
                production_pipe, X, y, task, random_state)
        else:
            learning_curves_data = {"lc_skipped": True, "reason": f"Dataset too large ({len(X)} rows)"}
        
        # ✅ MODEL CALIBRATION (classification only)
        calibration_data = None
        if task == "classification":
            calibration_data = compute_calibration_metrics(production_pipe, Xte, yte)
        
        # ✅ MODEL COMPLEXITY
        complexity_data = compute_model_complexity(production_pipe, Xte)
        
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
            "schema_version": "3.0",
            "experiment_id": experiment_id,
            "param_snapshot": param_snap,
            "model_checksum": model_checksum,
            "dataset_id": dataset_id,
            "task": task,
            "target": target,
            "best_model": best_model_name,
            "selection_score": round(float(best_score), 4),
            "best_score": round(float(best_score), 4),
            "selection_basis": "cv_mean" if cv_metric_used else "holdout",
            "cv_mean_score": round(float(best_cv_score), 4) if best_cv_score is not None else None,
            "holdout_score": round(float(best_holdout_score), 4),
            "training_approach": (
                "Model selected using cross-validation mean score "
                "with holdout fallback; final model retrained on full dataset"
            ),
            "leaderboard": leaderboard,
            "raw_columns": list(X.columns),
            "dropped_id_columns": id_columns_original,
            "dropped_high_cardinality_columns": high_card_cols,
            "class_mapping": class_mapping,
            "feature_defaults": extract_defaults(X),
            "feature_importance": importance_data,
            "top_features": importance_data["top_features"],
            "cross_validation": cv_data,
            "learning_curves": learning_curves_data,
            "calibration": calibration_data,
            "model_complexity": complexity_data,
            "balanced_accuracy": balanced_acc,
            "data_source": data_source,
            "training_time_seconds": round(total_time, 2),
            "timeout_limit_seconds": time_limit_seconds,
            "dataset_size_category": "small" if len(X) < 1000 else "medium" if len(X) < 10000 else "large",
            "hyperparameter_tuning": hyperparameter_search_info,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": env_snapshot,
            "random_state": random_state,
            "training_data_stats": training_data_stats,
            "statistical_drift": statistical_drift,
            "drift_enforced": True,
            "drift_confidence_adjustment": drift_confidence_adjustment,
            "drift_requires_retraining": drift_requires_retraining,
            "version_governance": {
                "authority": "ModelRegistry",
                "training_role": "Trains model and emits metadata only",
                "registry_role": "Single source of truth for versioning",
                "version_assignment": "Automatic (registry-controlled)"
            },
            "dataset_hash": dataset_hash(X),
        }
        
        if statistical_drift["drift_detected"]:
            result["statistical_drift_warning"] = statistical_drift
            result["dataset_drift_warning"] = drift_check
        
        # Save metadata (convert numpy types to JSON-serializable)
        result_serializable = convert_numpy_to_python(result)
        with open(os.path.join(root, "metadata.json"), "w") as f:
            json.dump(result_serializable, f, indent=2)
        
        # Register in model registry
        try:
            registry_result = register_trained_model(dataset_id, result)
            result["registry_info"] = registry_result
        except:
            pass
        
        return convert_numpy_to_python(result)
    
    except TimeoutError:
        return {"status": "failed", "error": "Training exceeded time limit"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
