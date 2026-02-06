import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_anomaly_detection(df: pd.DataFrame, features: list, method: str = "Isolation Forest", contamination: float = 0.05, random_state: int = 42) -> Dict[str, Any]:
    # Validate features
    # Try exact match first, then case-insensitive match
    available_features = []
    missing = []
    
    for f in features:
        if f in df.columns:
            available_features.append(f)
        else:
            # Try case-insensitive match
            matches = [col for col in df.columns if col.lower() == f.lower()]
            if matches:
                available_features.append(matches[0])
            else:
                missing.append(f)
    
    if missing or len(available_features) == 0:
        return {
            "status": "failed",
            "error": f"Features not found: {', '.join(missing if missing else features)}. Available features: {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}",
            "missing_features": missing,
            "available_features": df.columns.tolist()
        }
    
    scaler = StandardScaler()
    X = df[available_features].dropna()
    if X.empty:
        return {
            "status": "failed",
            "error": "No data available after dropping missing values."
        }
    X_scaled = scaler.fit_transform(X)
    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=random_state)
        labels = model.fit_predict(X_scaled)
    elif method == "One-Class SVM":
        model = OneClassSVM(nu=contamination)
        labels = model.fit_predict(X_scaled)
    elif method == "Local Outlier Factor":
        model = LocalOutlierFactor(contamination=contamination)
        labels = model.fit_predict(X_scaled)
    else:
        return {"status": "failed", "error": "Unknown anomaly detection method"}
    # 2D PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='coolwarm', alpha=0.7)
    ax.set_title(f"{method} Anomaly Detection (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Anomaly")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return {
        "status": "ok",
        "labels": labels.tolist(),
        "pca": X_pca.tolist(),
        "plot_base64": img_base64,
        "features_used": available_features,
        "n_samples": len(X),
        "n_anomalies": int((labels == -1).sum())
    }
