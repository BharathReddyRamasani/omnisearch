import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_clustering(df: pd.DataFrame, features: list, algo: str = "KMeans", n_clusters: int = 3, random_state: int = 42) -> Dict[str, Any]:
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
    if algo == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state)
    elif algo == "DBSCAN":
        model = DBSCAN()
    elif algo == "Agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algo == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
    else:
        return {"status": "failed", "error": "Unknown clustering algorithm"}
    labels = model.fit_predict(X_scaled)
    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='tab10', alpha=0.7)
    ax.set_title(f"{algo} Clustering (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return {
        "status": "ok",
        "labels": labels.tolist(),
        "pca": X_pca.tolist(),
        "plot_base64": img_base64
    }
