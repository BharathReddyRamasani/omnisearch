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





import os, json, time, joblib
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

from backend.services.utils import load_raw, model_dir


# =====================================================
# DEFAULT VALUE EXTRACTION
# =====================================================
def extract_defaults(X: pd.DataFrame):
    defaults = {}
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            defaults[c] = float(X[c].median())
        else:
            defaults[c] = X[c].mode().iloc[0]
    return defaults


# =====================================================
# TOP FEATURE EXTRACTION (RAW LEVEL)
# =====================================================
def extract_top_features(pipe, raw_columns, k=8):
    model = pipe.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        return raw_columns[:k]

    scores = model.feature_importances_
    ranked = sorted(
        zip(raw_columns, scores[:len(raw_columns)]),
        key=lambda x: x[1],
        reverse=True
    )
    return [f for f, _ in ranked[:k]]


# =====================================================
# MAIN TRAIN FUNCTION
# =====================================================
def train_model_logic(dataset_id: str, target: str):
    df = load_raw(dataset_id)

    if target not in df.columns:
        return {"status": "failed", "error": "Invalid target"}

    X = df.drop(columns=[target])
    y = df[target]

    mask = ~y.isna()
    X, y = X[mask], y[mask]

    if len(X) < 20:
        return {"status": "failed", "error": "Insufficient data"}

    # -------------------------------------------------
    # TASK DETECTION
    # -------------------------------------------------
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
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ])

    # -------------------------------------------------
    # MODEL ZOO (ENTERPRISE STANDARD)
    # -------------------------------------------------
    models = (
        {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(),
        }
        if task == "classification"
        else {
            "LinearRegression": LinearRegression(),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(),
        }
    )

    scorer = accuracy_score if task == "classification" else r2_score

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    leaderboard = []
    best_pipe = None
    best_score = -1e9
    best_model_name = None

    # -------------------------------------------------
    # TRAIN ALL MODELS
    # -------------------------------------------------
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

    # -------------------------------------------------
    # SAVE BEST MODEL
    # -------------------------------------------------
    root = model_dir(dataset_id)
    joblib.dump(best_pipe, os.path.join(root, "model.pkl"))

    top_features = extract_top_features(best_pipe, list(X.columns))

    # -------------------------------------------------
    # METADATA (STABLE CONTRACT)
    # -------------------------------------------------
    result = {
        "status": "ok",
        "task": task,
        "target": target,
        "best_model": best_model_name,
        "best_score": round(float(best_score), 4),
        "leaderboard": leaderboard,                # ✅ multi-model
        "raw_columns": list(X.columns),
        "feature_defaults": extract_defaults(X),
        "top_features": top_features,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(os.path.join(root, "metadata.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result
