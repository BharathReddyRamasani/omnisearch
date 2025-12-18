import pandas as pd
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

BASE_DATA_DIR = "data/datasets"

def load_df(dataset_id):
    path = os.path.join(BASE_DATA_DIR, dataset_id, "raw.csv")
    return pd.read_csv(path)

def run_eda(dataset_id):
    df = load_df(dataset_id)

    # Missing values
    missing = df.isnull().sum().to_dict()

    # Summary stats
    stats = df.describe(include="all").fillna("").to_dict()

    # Histograms (numeric only)
    plots = {}
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        ax.set_title(col)

        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)

        plots[col] = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "missing": missing,
        "stats": stats,
        "plots": plots
    }
