import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_pca(df: pd.DataFrame, features: list, n_components: int = 2, random_state: int = 42) -> Dict[str, Any]:
    # Validate features
    missing = [f for f in features if f not in df.columns]
    if missing:
        return {
            "status": "failed",
            "error": f"Features not found: {', '.join(missing)}",
            "missing_features": missing
        }
    scaler = StandardScaler()
    X = df[features].dropna()
    if X.empty:
        return {
            "status": "failed",
            "error": "No data available after dropping missing values."
        }
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7)
    ax.set_title("PCA Projection")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return {
        "status": "ok",
        "components": X_pca.tolist(),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "plot_base64": img_base64
    }

def run_tsne(df: pd.DataFrame, features: list, n_components: int = 2, random_state: int = 42) -> Dict[str, Any]:
    # Validate features
    missing = [f for f in features if f not in df.columns]
    if missing:
        return {
            "status": "failed",
            "error": f"Features not found: {', '.join(missing)}",
            "missing_features": missing
        }
    scaler = StandardScaler()
    X = df[features].dropna()
    if X.empty:
        return {
            "status": "failed",
            "error": "No data available after dropping missing values."
        }
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=n_components, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:,0], X_tsne[:,1], alpha=0.7)
    ax.set_title("t-SNE Projection")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return {
        "status": "ok",
        "components": X_tsne.tolist(),
        "plot_base64": img_base64
    }
