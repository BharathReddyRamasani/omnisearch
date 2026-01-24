"""
ENHANCED EDA WITH:
✅ Correlation heatmaps (JSON format)
✅ Distribution analysis
✅ Outlier detection (IQR, Z-score)
✅ Data quality metrics
✅ Downloadable report structure
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.services.utils import load_raw, load_clean


def _numeric_cols(df):
    """Get numeric column names"""
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def _categorical_cols(df):
    """Get categorical column names"""
    return df.select_dtypes(include=["object"]).columns.tolist()


def safe_numeric_summary(s: pd.Series) -> Dict:
    """Numeric column statistics"""
    if len(s) == 0:
        return {"mean": None, "std": 0, "min": None, "max": None}
    
    valid = s.dropna()
    if len(valid) == 0:
        return {"mean": None, "std": 0, "min": None, "max": None}
    
    return {
        "mean": round(float(valid.mean()), 4),
        "std": round(float(valid.std()), 4) if valid.std() > 0 else 0.0,
        "min": round(float(valid.min()), 4),
        "25%": round(float(valid.quantile(0.25)), 4),
        "median": round(float(valid.quantile(0.50)), 4),
        "75%": round(float(valid.quantile(0.75)), 4),
        "max": round(float(valid.max()), 4),
        "non_null_count": int(len(valid)),
        "null_count": int(s.isna().sum())
    }


def safe_categorical_summary(s: pd.Series) -> Dict:
    """Categorical column statistics"""
    valid = s.dropna()
    if len(valid) == 0:
        return {"unique": 0, "top": None, "freq": 0, "null_count": len(s)}
    
    value_counts = valid.value_counts()
    top_val = value_counts.index[0] if len(value_counts) > 0 else None
    top_freq = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
    
    return {
        "unique": int(valid.nunique()),
        "top": str(top_val) if top_val is not None else None,
        "freq": top_freq,
        "non_null_count": int(len(valid)),
        "null_count": int(s.isna().sum())
    }


def compute_correlation_matrix(df: pd.DataFrame) -> Dict:
    """
    Compute correlation matrix for numeric columns.
    Returns JSON-serializable format.
    """
    numeric = df.select_dtypes(include=[np.number])
    
    if len(numeric.columns) < 2:
        return {"correlation_matrix": {}, "note": "At least 2 numeric columns required"}
    
    try:
        corr = numeric.corr()
        
        # Convert to JSON-serializable format
        corr_dict = {}
        for col1 in corr.columns:
            corr_dict[str(col1)] = {}
            for col2 in corr.columns:
                corr_dict[str(col1)][str(col2)] = round(float(corr.loc[col1, col2]), 4)
        
        return {"correlation_matrix": corr_dict, "columns": list(numeric.columns)}
    except Exception as e:
        return {"error": str(e)}


def detect_outliers_iqr(df: pd.DataFrame) -> Dict:
    """
    Detect outliers using Interquartile Range method.
    """
    outliers = {}
    
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 4:
            continue
        
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            continue
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (s < lower_bound) | (s > upper_bound)
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outliers[col] = {
                "method": "IQR",
                "count": int(outlier_count),
                "percentage": round(float(outlier_count / len(df) * 100), 2),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4)
            }
    
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, threshold: float = 3.0) -> Dict:
    """
    Detect outliers using Z-score method.
    """
    outliers = {}
    
    for col in _numeric_cols(df):
        s = df[col].dropna()
        if len(s) < 2:
            continue
        
        if s.std() == 0:
            continue
        
        z_scores = np.abs((s - s.mean()) / s.std())
        outlier_mask = z_scores > threshold
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            outliers[col] = {
                "method": "Z-Score",
                "threshold": threshold,
                "count": int(outlier_count),
                "percentage": round(float(outlier_count / len(df) * 100), 2)
            }
    
    return outliers


def generate_eda(dataset_id: str) -> Dict:
    """
    Comprehensive EDA with visualization support.
    """
    try:
        # Prefer clean data
        try:
            df = load_clean(dataset_id)
            data_source = "clean"
        except:
            df = load_raw(dataset_id)
            data_source = "raw"
        
        # Basic metrics
        n_rows = len(df)
        n_cols = len(df.columns)
        
        # Missing data
        total_cells = n_rows * n_cols
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / max(1, total_cells)) * 100, 2)
        
        # Duplicates
        n_duplicates = int(df.duplicated().sum())
        dup_pct = round((n_duplicates / max(1, n_rows)) * 100, 2)
        
        # Data quality score
        quality_score = round(max(0, 100 - missing_pct - (dup_pct * 2)), 1)
        
        # Column summaries
        summary = {}
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                summary[col] = safe_numeric_summary(s)
            else:
                summary[col] = safe_categorical_summary(s)
        
        # Correlation matrix
        correlation = compute_correlation_matrix(df)
        
        # Outlier detection
        outliers_iqr = detect_outliers_iqr(df)
        outliers_zscore = detect_outliers_zscore(df, threshold=3.0)
        
        # Distribution analysis
        distributions = {}
        for col in _numeric_cols(df):
            s = df[col].dropna()
            if len(s) > 0:
                distributions[col] = {
                    "skewness": round(float(s.skew()), 4) if len(s) > 1 else 0,
                    "kurtosis": round(float(s.kurtosis()), 4) if len(s) > 1 else 0,
                    "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 4)
                }
        
        # Categorical analysis
        categorical_analysis = {}
        for col in _categorical_cols(df):
            s = df[col].dropna()
            if len(s) > 0:
                value_counts = s.value_counts().head(10)
                categorical_analysis[col] = {
                    "unique_count": int(s.nunique()),
                    "top_values": {
                        str(k): int(v) for k, v in value_counts.items()
                    }
                }
        
        # Data types
        dtypes = df.dtypes.astype(str).to_dict()
        
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "data_source": data_source,
            "overview": {
                "rows": n_rows,
                "columns": n_cols,
                "memory_usage_mb": round(float(df.memory_usage(deep=True).sum() / 1024 / 1024), 2),
                "quality_score": quality_score
            },
            "missing_data": {
                "total_missing": total_missing,
                "missing_percentage": missing_pct,
                "missing_by_column": df.isna().sum().to_dict()
            },
            "duplicates": {
                "duplicate_rows": n_duplicates,
                "duplicate_percentage": dup_pct
            },
            "column_summaries": summary,
            "dtypes": dtypes,
            "correlation": correlation,
            "distributions": distributions,
            "categorical_analysis": categorical_analysis,
            "outliers": {
                "iqr_method": outliers_iqr,
                "zscore_method": outliers_zscore,
                "total_outliers_iqr": sum(o["count"] for o in outliers_iqr.values())
            },
            "visualization_hints": {
                "heatmap_columns": list(correlation.get("columns", [])),
                "distribution_columns": list(distributions.keys()),
                "categorical_columns": list(categorical_analysis.keys()),
                "recommend_log_transform": [col for col in _numeric_cols(df) if distributions.get(col, {}).get("skewness", 0) > 1.5]
            }
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "EDA_FAILED"
        }


def generate_data_profile(dataset_id: str) -> Dict:
    """
    Generate downloadable data profile/report.
    """
    eda_result = generate_eda(dataset_id)
    
    if eda_result.get("status") != "ok":
        return eda_result
    
    # Create markdown-friendly report structure
    report = {
        "title": f"Data Profile Report - {dataset_id}",
        "generated_at": pd.Timestamp.now().isoformat(),
        "sections": {
            "overview": eda_result.get("overview", {}),
            "data_quality": {
                "missing_percentage": eda_result.get("missing_data", {}).get("missing_percentage", 0),
                "duplicate_percentage": eda_result.get("duplicates", {}).get("duplicate_percentage", 0),
                "quality_score": eda_result.get("overview", {}).get("quality_score", 0)
            },
            "column_details": eda_result.get("column_summaries", {}),
            "key_findings": {
                "high_cardinality_columns": [
                    col for col, info in eda_result.get("categorical_analysis", {}).items()
                    if info.get("unique_count", 0) > 50
                ],
                "columns_with_missing": [
                    col for col, count in eda_result.get("missing_data", {}).get("missing_by_column", {}).items()
                    if count > 0
                ],
                "columns_with_outliers": list(eda_result.get("outliers", {}).get("iqr_method", {}).keys()),
                "skewed_distributions": eda_result.get("visualization_hints", {}).get("recommend_log_transform", [])
            }
        }
    }
    
    return {
        "status": "ok",
        "report": report,
        "download_format": "JSON",
        "data_for_export": eda_result
    }
