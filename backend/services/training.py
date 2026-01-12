# # import os, json, joblib, time
# # import numpy as np
# # import pandas as pd
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.preprocessing import OneHotEncoder, StandardScaler
# # from sklearn.impute import SimpleImputer
# # from sklearn.ensemble import (
# #     RandomForestClassifier, RandomForestRegressor,
# #     GradientBoostingClassifier, GradientBoostingRegressor
# # )
# # from sklearn.linear_model import LogisticRegression, LinearRegression
# # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import accuracy_score, r2_score
# # from backend.services.utils import load_raw, model_dir


# # def extract_feature_importance(pipe, top_k=8):
# #     model = pipe.named_steps["model"]
# #     pre = pipe.named_steps["pre"]

# #     feature_names = []
# #     for name, transformer, cols in pre.transformers_:
# #         if name == "num":
# #             feature_names.extend(cols)
# #         elif name == "cat":
# #             ohe = transformer.named_steps["oh"]
# #             feature_names.extend(ohe.get_feature_names_out(cols))

# #     if hasattr(model, "feature_importances_"):
# #         scores = model.feature_importances_
# #     elif hasattr(model, "coef_"):
# #         scores = np.abs(model.coef_).ravel()
# #     else:
# #         return {}

# #     imp = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
# #     return {k: round(float(v), 6) for k, v in imp[:top_k]}


# # def extract_defaults(X: pd.DataFrame):
# #     defaults = {}
# #     for c in X.columns:
# #         if pd.api.types.is_numeric_dtype(X[c]):
# #             defaults[c] = float(X[c].median())
# #         else:
# #             defaults[c] = X[c].mode().iloc[0]
# #     return defaults


# # def train_model_logic(dataset_id: str, target: str):
# #     df = load_raw(dataset_id)
# #     if target not in df.columns:
# #         return {"status": "failed", "error": "Invalid target"}

# #     X = df.drop(columns=[target])
# #     y = df[target]

# #     X, y = X[~y.isna()], y[~y.isna()]
# #     if len(X) < 20:
# #         return {"status": "failed", "error": "Not enough valid rows"}

# #     task = "classification" if y.nunique() <= 15 else "regression"

# #     num = X.select_dtypes(include="number").columns.tolist()
# #     cat = X.select_dtypes(include="object").columns.tolist()

# #     pre = ColumnTransformer([
# #         ("num", Pipeline([
# #             ("imp", SimpleImputer(strategy="median")),
# #             ("sc", StandardScaler())
# #         ]), num),
# #         ("cat", Pipeline([
# #             ("imp", SimpleImputer(strategy="most_frequent")),
# #             ("oh", OneHotEncoder(handle_unknown="ignore"))
# #         ]), cat),
# #     ])

# #     models = (
# #         {
# #             "LogisticRegression": LogisticRegression(max_iter=1000),
# #             "DecisionTree": DecisionTreeClassifier(),
# #             "RandomForest": RandomForestClassifier(n_estimators=200),
# #             "GradientBoosting": GradientBoostingClassifier(),
# #         } if task == "classification" else {
# #             "LinearRegression": LinearRegression(),
# #             "DecisionTree": DecisionTreeRegressor(),
# #             "RandomForest": RandomForestRegressor(n_estimators=200),
# #             "GradientBoosting": GradientBoostingRegressor(),
# #         }
# #     )

# #     scorer = accuracy_score if task == "classification" else r2_score
# #     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# #     leaderboard, best_pipe, best_score, best_model = [], None, -1e9, None

# #     for name, model in models.items():
# #         pipe = Pipeline([("pre", pre), ("model", model)])
# #         pipe.fit(Xtr, ytr)
# #         score = scorer(yte, pipe.predict(Xte))

# #         leaderboard.append({
# #             "model": name,
# #             "score": round(float(score), 4),
# #             "train_rows": len(Xtr),
# #             "test_rows": len(Xte),
# #         })

# #         if score > best_score:
# #             best_score, best_pipe, best_model = score, pipe, name

# #     root = model_dir(dataset_id)
# #     joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

# #     result = {
# #         "status": "ok",
# #         "task": task,
# #         "target": target,
# #         "best_model": best_model,
# #         "best_score": round(float(best_score), 4),
# #         "leaderboard": leaderboard,
# #         "feature_defaults": extract_defaults(X),
# #         "feature_importance": extract_feature_importance(best_pipe, top_k=12),
# #         "top_features": list(extract_feature_importance(best_pipe, 8).keys()),
# #         "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
# #     }

# #     with open(os.path.join(root, "metadata.json"), "w") as f:
# #         json.dump(result, f, indent=2)

# #     return result

# import os, json, time, joblib
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import (
#     RandomForestClassifier, RandomForestRegressor,
#     GradientBoostingClassifier, GradientBoostingRegressor
# )
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, r2_score
from backend.services.utils import load_raw, model_dir
from backend.services.model_registry import ModelRegistry

# # -----------------------------
# # FEATURE IMPORTANCE (RAW LEVEL)
# # -----------------------------
# def extract_feature_importance(pipe, top_k=12):
#     model = pipe.named_steps["model"]
#     pre = pipe.named_steps["pre"]

#     feature_names = []
#     for name, transformer, cols in pre.transformers_:
#         if name == "num":
#             feature_names.extend(cols)
#         elif name == "cat":
#             ohe = transformer.named_steps["oh"]
#             feature_names.extend(ohe.get_feature_names_out(cols))

#     if hasattr(model, "feature_importances_"):
#         scores = model.feature_importances_
#     elif hasattr(model, "coef_"):
#         scores = np.abs(model.coef_).ravel()
#     else:
#         return {}

#     ranked = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
#     return {k: round(float(v), 6) for k, v in ranked[:top_k]}


# # -----------------------------
# # DEFAULT VALUE EXTRACTION
# # -----------------------------
# def extract_defaults(X: pd.DataFrame):
#     defaults = {}
#     for c in X.columns:
#         if pd.api.types.is_numeric_dtype(X[c]):
#             defaults[c] = float(X[c].median())
#         else:
#             defaults[c] = X[c].mode().iloc[0]
#     return defaults


# # -----------------------------
# # MAIN TRAIN FUNCTION
# # -----------------------------
# def train_model_logic(dataset_id: str, target: str):
#     df = load_raw(dataset_id)

#     if target not in df.columns:
#         return {"status": "failed", "error": "Invalid target"}

#     X = df.drop(columns=[target])
#     y = df[target]
#     mask = ~y.isna()
#     X, y = X[mask], y[mask]

#     if len(X) < 20:
#         return {"status": "failed", "error": "Not enough valid rows"}

#     task = "classification" if y.nunique() <= 15 else "regression"

#     num = X.select_dtypes(include="number").columns.tolist()
#     cat = X.select_dtypes(include="object").columns.tolist()

#     pre = ColumnTransformer([
#         ("num", Pipeline([
#             ("imp", SimpleImputer(strategy="median")),
#             ("sc", StandardScaler())
#         ]), num),
#         ("cat", Pipeline([
#             ("imp", SimpleImputer(strategy="most_frequent")),
#             ("oh", OneHotEncoder(handle_unknown="ignore"))
#         ]), cat),
#     ])

#     models = (
#         {
#             "LogisticRegression": LogisticRegression(max_iter=1000),
#             "DecisionTree": DecisionTreeClassifier(),
#             "RandomForest": RandomForestClassifier(n_estimators=200),
#             "GradientBoosting": GradientBoostingClassifier(),
#         }
#         if task == "classification"
#         else {
#             "LinearRegression": LinearRegression(),
#             "DecisionTree": DecisionTreeRegressor(),
#             "RandomForest": RandomForestRegressor(n_estimators=200),
#             "GradientBoosting": GradientBoostingRegressor(),
#         }
#     )

#     scorer = accuracy_score if task == "classification" else r2_score
#     Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

#     leaderboard = []
#     best_pipe, best_score, best_model = None, -1e9, None

#     for name, model in models.items():
#         pipe = Pipeline([("pre", pre), ("model", model)])
#         pipe.fit(Xtr, ytr)
#         score = scorer(yte, pipe.predict(Xte))

#         leaderboard.append({
#             "model": name,
#             "score": round(float(score), 4),
#             "train_rows": int(len(Xtr)),
#             "test_rows": int(len(Xte)),
#         })

#         if score > best_score:
#             best_score, best_pipe, best_model = score, pipe, name

#     root = model_dir(dataset_id)
#     joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

#     feature_importance = extract_feature_importance(best_pipe, top_k=12)

#     result = {
#         "status": "ok",
#         "task": task,
#         "target": target,
#         "best_model": best_model,
#         "best_score": round(float(best_score), 4),
#         "leaderboard": leaderboard,              # ✅ STABLE
#         "feature_importance": feature_importance,
#         "top_features": list(feature_importance.keys()),  # ✅ ALWAYS EXISTS
#         "feature_defaults": extract_defaults(X),           # ✅ ALWAYS EXISTS
#         "raw_columns": list(X.columns),
#         "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
#     }

#     with open(os.path.join(root, "metadata.json"), "w") as f:
#         json.dump(result, f, indent=2)

#     return result

import os, json, time, joblib, hashlib, signal
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
)

from backend.services.utils import load_raw, load_clean, model_dir
from backend.services.model_registry import ModelRegistry,register_trained_model


# =====================================================
# DATASET DRIFT DETECTION
# =====================================================
def detect_dataset_drift(dataset_id: str, current_df: pd.DataFrame) -> Dict:
    """Detect if dataset has changed since model training"""
    registry = ModelRegistry.load_registry()
    
    if dataset_id not in registry["models"]:
        return {"drift_detected": False, "reason": "No previous model"}
    
    active_version = ModelRegistry.get_active_version(dataset_id)
    if not active_version:
        return {"drift_detected": False, "reason": "No active model"}
    
    stored_hash = active_version["metadata"].get("dataset_hash")
    if not stored_hash:
        return {"drift_detected": False, "reason": "No hash stored"}
    
    current_hash = dataset_hash(current_df)
    
    if current_hash != stored_hash:
        # Analyze what changed
        stored_metadata = active_version["metadata"]
        stored_columns = set(stored_metadata.get("raw_columns", []))
        current_columns = set(current_df.columns)
        
        added_columns = current_columns - stored_columns
        removed_columns = stored_columns - current_columns
        common_columns = stored_columns & current_columns
        
        # Check for schema changes in common columns
        schema_changes = []
        for col in common_columns:
            stored_dtype = str(stored_metadata.get("feature_defaults", {}).get(col, "unknown"))
            current_dtype = str(current_df[col].dtype)
            if stored_dtype != current_dtype:
                schema_changes.append({
                    "column": col,
                    "old_type": stored_dtype,
                    "new_type": current_dtype
                })
        
        return {
            "drift_detected": True,
            "reason": "Dataset hash mismatch",
            "stored_hash": stored_hash,
            "current_hash": current_hash,
            "changes": {
                "added_columns": list(added_columns),
                "removed_columns": list(removed_columns),
                "schema_changes": schema_changes,
                "row_count_change": len(current_df) - stored_metadata.get("leaderboard", [{}])[0].get("train_rows", 0) - stored_metadata.get("leaderboard", [{}])[0].get("test_rows", 0)
            },
            "recommendation": "Retrain model with current dataset",
            "can_retrain": True
        }
    
    return {"drift_detected": False, "reason": "Dataset unchanged"}


# =====================================================
# TRAINING CONFIGURATION (INDUSTRIAL GRADE)
# =====================================================
class TrainingConfig:
    """Production-ready training configuration with scalability guards"""

    @staticmethod
    def get_model_config(n_samples: int, n_features: int, task: str):
        """Dataset size-aware model selection and hyperparameter tuning"""

        # Dataset size categories
        if n_samples < 1000:
            size_category = "small"
        elif n_samples < 10000:
            size_category = "medium"
        elif n_samples < 100000:
            size_category = "large"
        else:
            size_category = "xlarge"

        # Base configurations
        configs = {
            "classification": {
                "small": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 10, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 50,
                        "max_depth": 10,
                        "max_samples": min(0.8, max(0.1, 1000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 50,
                        "max_depth": 3,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                    "NaiveBayes": {},
                    "SVM": {"random_state": 42, "max_iter": 1000},
                    "KNN": {"n_neighbors": min(5, max(1, int(np.sqrt(n_samples))))},
                },
                "medium": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 15, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 100,
                        "max_depth": 15,
                        "max_samples": min(0.8, max(0.1, 5000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 100,
                        "max_depth": 4,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                    "NaiveBayes": {},
                    "SVM": {"random_state": 42, "max_iter": 2000},
                    "KNN": {"n_neighbors": min(7, max(1, int(np.sqrt(n_samples))))},
                },
                "large": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 20, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 200,
                        "max_depth": 20,
                        "max_samples": min(0.8, max(0.1, 10000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 200,
                        "max_depth": 5,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                    "NaiveBayes": {},
                    "SVM": {"random_state": 42, "max_iter": 5000},
                    "KNN": {"n_neighbors": min(9, max(1, int(np.sqrt(n_samples))))},
                },
                "xlarge": {
                    "LogisticRegression": {"max_iter": 1000, "random_state": 42},
                    "DecisionTree": {"max_depth": 25, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 300,
                        "max_depth": 25,
                        "max_samples": min(0.8, max(0.1, 20000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 300,
                        "max_depth": 6,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                }
            },
            "regression": {
                "small": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 10, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 50,
                        "max_depth": 10,
                        "max_samples": min(0.8, max(0.1, 1000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 50,
                        "max_depth": 3,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                },
                "medium": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 15, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 100,
                        "max_depth": 15,
                        "max_samples": min(0.8, max(0.1, 5000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 100,
                        "max_depth": 4,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                },
                "large": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 20, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 200,
                        "max_depth": 20,
                        "max_samples": min(0.8, max(0.1, 10000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 200,
                        "max_depth": 5,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                },
                "xlarge": {
                    "LinearRegression": {},
                    "Ridge": {"random_state": 42},
                    "Lasso": {"random_state": 42},
                    "DecisionTree": {"max_depth": 25, "random_state": 42},
                    "RandomForest": {
                        "n_estimators": 300,
                        "max_depth": 25,
                        "max_samples": min(0.8, max(0.1, 20000/n_samples)),
                        "random_state": 42,
                        "n_jobs": -1
                    },
                    "GradientBoosting": {
                        "n_estimators": 300,
                        "max_depth": 6,
                        "validation_fraction": 0.1,
                        "n_iter_no_change": 5,
                        "random_state": 42
                    },
                }
            }
        }

        return configs[task][size_category]

    @staticmethod
    def should_skip_polynomial(n_samples: int, n_features: int) -> bool:
        """Skip polynomial regression for large datasets to prevent explosion"""
        # Skip if dataset is large or has many features
        return n_samples > 5000 or n_features > 20

    @staticmethod
    def get_timeout_seconds(n_samples: int) -> int:
        """Adaptive timeout based on dataset size"""
        if n_samples < 1000:
            return 60  # 1 minute
        elif n_samples < 10000:
            return 300  # 5 minutes
        elif n_samples < 50000:
            return 900  # 15 minutes
        else:
            return 1800  # 30 minutes


# =====================================================
# TIMEOUT HANDLER (WINDOWS-COMPATIBLE)
# =====================================================
class TimeoutError(Exception):
    pass

# Windows doesn't support signal.SIGALRM, so we use a no-op approach
# The ThreadPoolExecutor in registry.py handles job timeouts
def set_timeout(seconds: int):
    """Set timeout - no-op on Windows, uses signal on Unix"""
    import platform
    if platform.system() != "Windows":
        import signal
        signal.alarm(seconds)

def cancel_timeout():
    """Cancel timeout - no-op on Windows, uses signal on Unix"""
    import platform
    if platform.system() != "Windows":
        import signal
        signal.alarm(0)


# =====================================================
# DATASET HASH
# =====================================================
def dataset_hash(df: pd.DataFrame) -> str:
    df_sorted = df[sorted(df.columns)]
    return hashlib.md5(pd.util.hash_pandas_object(df_sorted, index=True).values).hexdigest()


# =====================================================
# ID COLUMN DETECTION (INDUSTRIAL GRADE)
# =====================================================
def detect_id_columns(df: pd.DataFrame, uniqueness_threshold: float = 0.95, min_rows: int = 100) -> list[str]:
    if len(df) == 0:
        return []
    
    n_rows = len(df)
    threshold = max(uniqueness_threshold, 0.99 if n_rows < min_rows else uniqueness_threshold)
    
    id_candidates = []
    suspicious_names = {"id", "uuid", "key", "code", "guid", "sk", "pk"}
    
    for col in df.columns:
        col_lower = col.lower()
        
        if col_lower in {"id", "uuid", "guid"} or col_lower.endswith("_id") or col_lower.endswith("_uuid"):
            id_candidates.append(col)
            continue
        
        if any(pattern in col_lower for pattern in suspicious_names):
            uniqueness = df[col].nunique() / n_rows
            if uniqueness >= threshold:
                id_candidates.append(col)
    
    return id_candidates


# =====================================================
# DEFAULT VALUE EXTRACTION
# =====================================================
def extract_defaults(X: pd.DataFrame):
    defaults = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            defaults[c] = float(X[c].median())
        else:
            defaults[c] = X[c].mode().iloc[0] if not X[c].mode().empty else "UNKNOWN"
    return defaults


# =====================================================
# FEATURE IMPORTANCE EXTRACTION (ROBUST)
# =====================================================
def extract_feature_importance(pipe, raw_columns, task: str):
    """Extract feature importance with fallback strategies"""
    model = pipe.named_steps["model"]
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
            except Exception:
                feature_names.extend(cols)
    
    # Handle polynomial features
    if "poly_model" in pipe.named_steps:
        poly = pipe.named_steps["poly_model"].named_steps["poly"]
        poly_features = poly.get_feature_names_out(feature_names)
        feature_names = poly_features
    
    # Extract importance scores based on model type
    importance_supported = True
    importance_method = None
    
    if hasattr(model, "feature_importances_"):
        # Tree-based models
        scores = model.feature_importances_
        importance_method = "feature_importances_"
    elif hasattr(model, "coef_"):
        # Linear models
        coef = model.coef_.ravel()
        if task == "classification" and len(np.unique(model.classes_)) == 2:
            # Binary classification - use absolute coefficients
            scores = np.abs(coef)
        else:
            # Multi-class or regression - use absolute coefficients
            scores = np.abs(coef)
        importance_method = "coefficients"
    else:
        # No importance available
        importance_supported = False
        scores = np.ones(len(feature_names)) / len(feature_names)  # Equal weights
        importance_method = "uniform_fallback"
    
    if len(scores) != len(feature_names):
        # Mismatch - fallback to uniform
        importance_supported = False
        scores = np.ones(len(feature_names)) / len(feature_names)
        importance_method = "uniform_fallback"
    
    # Create importance dictionary
    importance_dict = {}
    for name, score in zip(feature_names, scores):
        # Group by original feature name (before encoding)
        base_name = name.split("_")[0] if "_" in name else name
        if base_name not in importance_dict:
            importance_dict[base_name] = []
        importance_dict[base_name].append(float(score))
    
    # Aggregate by taking mean for each base feature
    aggregated = {k: np.mean(v) for k, v in importance_dict.items()}
    
    # Sort by importance
    ranked = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "importance_scores": dict(ranked),
        "importance_supported": importance_supported,
        "importance_method": importance_method,
        "top_features": [name for name, _ in ranked[:8]],
        "raw_feature_count": len(raw_columns),
        "processed_feature_count": len(feature_names)
    }


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def train_model_logic(dataset_id: str, target: str, test_size: float = 0.2, random_state: int = 42,
                     train_regression: bool = True, train_classification: bool = True,
                     time_limit_seconds: int = None):
    try:
        start_time = time.time()

        # Load data
        try:
            df = load_clean(dataset_id)
            data_source = "clean"
        except Exception:
            df = load_raw(dataset_id)
            data_source = "raw"

        if target not in df.columns:
            return {"status": "failed", "error": "Invalid target"}

        # Prepare data
        df_filtered = df
        X = df.drop(columns=[target])
        y = df[target]

        # Remove rows with missing target
        X, y = X[~y.isna()], y[~y.isna()]

        # Detect and drop ID columns
        id_columns = []
        for col in X.columns:
            if X[col].nunique() == len(X):  # Unique values = likely ID
                id_columns.append(col)
                X = X.drop(columns=[col])

        # Check for dataset drift before training
        drift_check = detect_dataset_drift(dataset_id, df_filtered)

        if len(X) < 20:
            return {"status": "failed", "error": "Insufficient data"}

        # Adaptive timeout
        if time_limit_seconds is None:
            time_limit_seconds = TrainingConfig.get_timeout_seconds(len(X))

        # Set timeout for this specific run (Windows-compatible)
        set_timeout(time_limit_seconds)

        task = "classification" if y.nunique() <= 15 else "regression"

        num = X.select_dtypes(include="number").columns.tolist()
        cat = X.select_dtypes(include="object").columns.tolist()

        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), num),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat),
        ])

        # Get dataset-aware model configurations
        model_configs = TrainingConfig.get_model_config(len(X), len(X.columns), task)

        # Build models dictionary based on task and flags
        models = {}

        if task == "classification" and train_classification:
            models.update({
                "LogisticRegression": LogisticRegression(**model_configs.get("LogisticRegression", {})),
                "DecisionTree": DecisionTreeClassifier(**model_configs.get("DecisionTree", {})),
                "RandomForest": RandomForestClassifier(**model_configs.get("RandomForest", {})),
                "GradientBoosting": GradientBoostingClassifier(**model_configs.get("GradientBoosting", {})),
                "NaiveBayes": GaussianNB(**model_configs.get("NaiveBayes", {})),
                "SVM": SVC(**model_configs.get("SVM", {})),
                "KNN": KNeighborsClassifier(**model_configs.get("KNN", {})),
            })
        elif task == "regression" and train_regression:
            models.update({
                "LinearRegression": LinearRegression(**model_configs.get("LinearRegression", {})),
                "Ridge": Ridge(**model_configs.get("Ridge", {})),
                "Lasso": Lasso(**model_configs.get("Lasso", {})),
                "DecisionTree": DecisionTreeRegressor(**model_configs.get("DecisionTree", {})),
                "RandomForest": RandomForestRegressor(**model_configs.get("RandomForest", {})),
                "GradientBoosting": GradientBoostingRegressor(**model_configs.get("GradientBoosting", {})),
            })

            # Conditionally add polynomial regression (skip for large datasets)
            if not TrainingConfig.should_skip_polynomial(len(X), len(X.columns)):
                models["PolynomialRegression"] = Pipeline([
                    ("poly", PolynomialFeatures(degree=2)),
                    ("linear", LinearRegression())
                ])

        if not models:
            return {"status": "failed", "error": f"No models selected for {task} task"}

        scorer = accuracy_score if task == "classification" else r2_score

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

        leaderboard = []
        best_pipe = None
        best_score = -1e9
        best_model_name = None
        model_details = []

        for name, model in models.items():
            model_start_time = time.time()

            try:
                if name == "PolynomialRegression":
                    # Polynomial is already a pipeline
                    pipe = Pipeline([
                        ("pre", pre),
                        ("poly_model", model)
                    ])
                else:
                    pipe = Pipeline([
                        ("pre", pre),
                        ("model", model)
                    ])

                pipe.fit(Xtr, ytr)
                preds = pipe.predict(Xte)
                score = scorer(yte, preds)

                training_time = time.time() - model_start_time

                # Additional metrics
                if task == "classification":
                    cm = confusion_matrix(yte, preds).tolist()
                    report = classification_report(yte, preds, output_dict=True, zero_division=0)
                    precision, recall, f1, _ = precision_recall_fscore_support(yte, preds, average='weighted', zero_division=0)
                    try:
                        auc = roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1]) if len(np.unique(yte)) == 2 else None
                    except:
                        auc = None
                    metrics = {
                        "accuracy": round(float(score), 4),
                        "precision": round(float(precision), 4),
                        "recall": round(float(recall), 4),
                        "f1_score": round(float(f1), 4),
                        "auc": round(float(auc), 4) if auc else None,
                        "confusion_matrix": cm
                    }
                else:
                    mae = mean_absolute_error(yte, preds)
                    mse = mean_squared_error(yte, preds)
                    rmse = np.sqrt(mse)
                    metrics = {
                        "r2_score": round(float(score), 4),
                        "mae": round(float(mae), 4),
                        "mse": round(float(mse), 4),
                        "rmse": round(float(rmse), 4)
                    }

                leaderboard.append({
                    "model": name,
                    "score": round(float(score), 4),
                    "train_rows": int(len(Xtr)),
                    "test_rows": int(len(Xte)),
                    "training_time_seconds": round(training_time, 2),
                    "metrics": metrics
                })

                model_details.append({
                    "model": name,
                    "score": round(float(score), 4),
                    "training_time_seconds": round(training_time, 2),
                    "metrics": metrics,
                    "predictions_sample": preds[:10].tolist() if len(preds) > 10 else preds.tolist()
                })

                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    best_model_name = name

            except Exception as e:
                # Log failed model but continue
                leaderboard.append({
                    "model": name,
                    "score": None,
                    "error": str(e),
                    "training_time_seconds": round(time.time() - model_start_time, 2)
                })

        # Cancel timeout alarm
        cancel_timeout()

        if best_pipe is None:
            return {"status": "failed", "error": "All models failed to train"}

        root = model_dir(dataset_id)
        joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

        # Top features based on final modeling columns
        feature_importance_data = extract_feature_importance(best_pipe, list(X.columns), task)
        top_features = feature_importance_data["top_features"]

        total_training_time = time.time() - start_time

        result = {
            "status": "ok",
            "schema_version": "1.0",
            "model_version": "v1",
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash(df_filtered),
            "task": task,
            "target": target,
            "best_model": best_model_name,
            "best_score": round(float(best_score), 4),
            "leaderboard": leaderboard,
            "model_details": model_details,
            "raw_columns": list(X.columns),           # Features actually used in model
            "dropped_id_columns": id_columns,         # ← NEW: audit trail
            "feature_defaults": extract_defaults(X),
            "feature_importance": feature_importance_data,
            "top_features": top_features,
            "data_source": data_source,
            "training_time_seconds": round(total_training_time, 2),
            "timeout_limit_seconds": time_limit_seconds,
            "dataset_size_category": "small" if len(X) < 1000 else "medium" if len(X) < 10000 else "large" if len(X) < 100000 else "xlarge",
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        # Add drift warning if detected
        if drift_check["drift_detected"]:
            result["dataset_drift_warning"] = drift_check

        with open(os.path.join(root, "metadata.json"), "w") as f:
            json.dump(result, f, indent=2)

        # Register the trained model in the registry
        registry_result = register_trained_model(dataset_id, result)
        result["registry_info"] = registry_result

        return result

    except TimeoutError as e:
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}




