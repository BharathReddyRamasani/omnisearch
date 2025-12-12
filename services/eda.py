# services/eda.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64, io
import json

plt.switch_backend("Agg")  # safe non-GUI backend

def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64  # plain base64 string, no data:image/... prefix

def safe_to_python(obj):
    """Convert numpy / pandas scalars to native python for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj

def hist_plot(series):
    # attempt coercion to numeric; if fails skip
    try:
        ser = pd.to_numeric(series.dropna(), errors="coerce")
        if ser.dropna().empty:
            # nothing numeric to plot
            return None
        plt.figure(figsize=(4,3))
        plt.hist(ser, bins=30)
        plt.title(series.name)
        return fig_to_base64()
    except Exception:
        return None

def corr_plot(df_numeric):
    try:
        plt.figure(figsize=(6,5))
        corr = df_numeric.corr()
        if corr.empty:
            return None
        sns.heatmap(corr, annot=False, cmap="Blues")
        plt.title("Correlation Heatmap")
        return fig_to_base64()
    except Exception:
        return None

def generate_eda(df):
    eda = {}

    # Missing values (convert to int)
    eda["missing"] = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}

    # Data types
    eda["dtypes"] = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}

    # Stats (safe): convert numeric-like describe to python types
    try:
        desc = df.describe(include="all").to_dict()
        safe_desc = {}
        for col, measures in desc.items():
            safe_measures = {}
            for mname, val in measures.items():
                safe_measures[str(mname)] = safe_to_python(val)
            safe_desc[str(col)] = safe_measures
        eda["stats"] = safe_desc
    except Exception:
        eda["stats"] = {}

    # Histograms for numeric columns
    hists = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        b64 = hist_plot(df[col])
        hists[str(col)] = b64
    eda["hists"] = hists

    # Correlation heatmap (numeric only)
    if len(numeric_cols) >= 2:
        try:
            eda["corr"] = corr_plot(df[numeric_cols])
        except Exception:
            eda["corr"] = None
    else:
        eda["corr"] = None

    return eda
