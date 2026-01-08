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
# from backend.services.utils import load_raw, model_dir


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

import os, json, time, joblib, hashlib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from backend.services.utils import load_raw, load_clean, model_dir


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
# TOP FEATURE EXTRACTION
# =====================================================
def extract_top_features(pipe, raw_columns, k=8):
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    
    if not hasattr(model, "feature_importances_"):
        return raw_columns[:k]
    
    importances = model.feature_importances_
    feature_names = []
    
    for name, transformer, cols in pre.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["oh"]
            feature_names.extend(ohe.get_feature_names_out(cols))
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    
    df_imp["raw_feature"] = df_imp["feature"].apply(lambda x: x.split("_")[0] if "_" in x else x)
    
    ranked = (
        df_imp.groupby("raw_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
    )
    
    return ranked.head(k).index.tolist()


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def train_model_logic(dataset_id: str, target: str):
    try:
        df = load_clean(dataset_id)
        data_source = "clean"
    except Exception:
        df = load_raw(dataset_id)
        data_source = "raw"

    if target not in df.columns:
        return {"status": "failed", "error": "Invalid target"}

    # Detect ID columns BEFORE dropping target
    id_columns = detect_id_columns(df)

    # Drop target and IDs for modeling
    drop_cols = [target] + id_columns
    invalid_drops = [c for c in drop_cols if c not in df.columns]
    if invalid_drops:
        # Safety: only drop what exists
        drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target]

    mask = ~y.isna()
    X, y = X[mask], y[mask]
    df_filtered = df[mask]  # For hashing — includes target + IDs

    if len(X) < 20:
        return {"status": "failed", "error": "Insufficient data"}

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

    models = (
        {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(n_estimators=100,max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(),
        }
        if task == "classification"
        else {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(n_estimators=100,max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(),
        }
    )

    scorer = accuracy_score if task == "classification" else r2_score

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    leaderboard = []
    best_pipe = None
    best_score = -1e9
    best_model_name = None

    for name, model in models.items():
        pipe = Pipeline([
            ("pre", pre),
            ("model", model)
        ])

        pipe.fit(Xtr, ytr)
        preds = pipe.predict(Xte)
        score = scorer(yte, preds)

        leaderboard.append({
            "model": name,
            "score": round(float(score), 4),
            "train_rows": int(len(Xtr)),
            "test_rows": int(len(Xte)),
        })

        if score > best_score:
            best_score = score
            best_pipe = pipe
            best_model_name = name

    root = model_dir(dataset_id)
    joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

    # Top features based on final modeling columns
    top_features = extract_top_features(best_pipe, list(X.columns))

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
        "raw_columns": list(X.columns),           # Features actually used in model
        "dropped_id_columns": id_columns,         # ← NEW: audit trail
        "feature_defaults": extract_defaults(X),
        "top_features": top_features,
        "data_source": data_source,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


