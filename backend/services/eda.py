"""
ENHANCED EDA WITH:
✅ Correlation heatmaps (JSON + base64 images)
✅ Distribution analysis (histograms, boxplots)
✅ Outlier detection (IQR, Z-score)
✅ Data quality metrics
✅ Downloadable HTML & JSON reports
✅ Column selection for wide datasets (>150 cols)
✅ Sample vs full EDA toggle
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO

# Configure matplotlib for non-GUI backend (prevents threading issues)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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


# =====================================================
# VISUALIZATION GENERATORS (BASE64 ENCODED)
# =====================================================
def generate_histogram_base64(series: pd.Series, column_name: str, bins: int = 30) -> str:
    """Generate histogram as base64 PNG"""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series.dropna(), bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Distribution: {column_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64
    except Exception as e:
        return ""


def generate_boxplot_base64(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    """Generate boxplot for numeric columns as base64 PNG"""
    try:
        if not numeric_cols:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        df[numeric_cols[:20]].boxplot(ax=ax)  # Limit to 20 cols for readability
        ax.set_title('Boxplot: Numeric Columns', fontsize=14, fontweight='bold')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64
    except Exception as e:
        return ""


def generate_heatmap_base64(df: pd.DataFrame) -> str:
    """Generate correlation heatmap as base64 PNG"""
    try:
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) < 2:
            return ""
        
        # Limit to top 20 columns for readability
        if len(numeric.columns) > 20:
            numeric = numeric.iloc[:, numeric.var().nlargest(20).index]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = numeric.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=80, bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_base64
    except Exception as e:
        return ""


def select_columns_for_eda(df: pd.DataFrame, max_cols: int = 50) -> Dict[str, Any]:
    """
    Intelligently select columns for wide datasets (>150 cols).
    Prioritizes numeric and categorical columns based on variance/cardinality.
    """
    if len(df.columns) <= max_cols:
        return {
            "numeric": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical": df.select_dtypes(include=['object']).columns.tolist()
        }
    
    # Select top numeric by variance
    numeric = df.select_dtypes(include=[np.number])
    numeric_selected = numeric.var().nlargest(min(30, len(numeric.columns))).index.tolist()
    
    # Select top categorical by cardinality
    categorical = df.select_dtypes(include=['object'])
    cat_selected = categorical.nunique().nlargest(min(20, len(categorical.columns))).index.tolist()
    
    return {
        "numeric": numeric_selected,
        "categorical": cat_selected,
        "total_columns": len(df.columns),
        "selected_columns": len(numeric_selected) + len(cat_selected),
        "selection_method": "variance (numeric) + cardinality (categorical)"
    }


def generate_html_eda_report(eda_result: Dict, dataset_id: str) -> str:
    """Generate downloadable HTML EDA report"""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - {dataset_id}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c5364;
            border-bottom: 3px solid #2c5364;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #203a43;
            margin-top: 30px;
            border-left: 4px solid #2c5364;
            padding-left: 10px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #2c5364;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .key-finding {{
            background-color: #fffacd;
            border-left: 4px solid #ffa500;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .generated-at {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 EDA Report - {dataset_id}</h1>
        
        <h2>📈 Executive Summary</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{eda_result.get("overview", {}).get("rows", 0):,}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{eda_result.get("overview", {}).get("columns", 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Quality Score</div>
                <div class="metric-value">{eda_result.get("overview", {}).get("quality_score", 0):.0f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{eda_result.get("missing_data", {}).get("missing_percentage", 0):.1f}%</div>
            </div>
        </div>
        
        <h2>🔍 Data Quality Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Cells</td>
                <td>{eda_result.get("overview", {}).get("rows", 0) * eda_result.get("overview", {}).get("columns", 0):,}</td>
            </tr>
            <tr>
                <td>Missing Values</td>
                <td>{eda_result.get("missing_data", {}).get("total_missing", 0):,}</td>
            </tr>
            <tr>
                <td>Duplicate Rows</td>
                <td>{eda_result.get("duplicates", {}).get("duplicate_rows", 0)}</td>
            </tr>
            <tr>
                <td>Memory Usage (MB)</td>
                <td>{eda_result.get("overview", {}).get("memory_usage_mb", 0):.2f}</td>
            </tr>
        </table>
        
        <h2>⚠️ Key Findings</h2>
        {generate_key_findings_html(eda_result)}
        
        <div class="generated-at">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
    return html_content


def generate_key_findings_html(eda_result: Dict) -> str:
    """Generate key findings HTML"""
    findings = []
    
    # High missing
    if eda_result.get("missing_data", {}).get("missing_percentage", 0) > 30:
        findings.append(f'<div class="key-finding">⚠️ High missing data: {eda_result["missing_data"]["missing_percentage"]:.1f}%</div>')
    
    # High duplicates
    if eda_result.get("duplicates", {}).get("duplicate_percentage", 0) > 5:
        findings.append(f'<div class="key-finding">⚠️ Duplicate rows detected: {eda_result["duplicates"]["duplicate_percentage"]:.1f}%</div>')
    
    # Outliers
    outlier_count = eda_result.get("outliers", {}).get("total_outliers_iqr", 0)
    if outlier_count > 0:
        findings.append(f'<div class="key-finding">⚠️ Outliers detected: {outlier_count} values</div>')
    
    # Skewed distributions
    skewed = eda_result.get("visualization_hints", {}).get("recommend_log_transform", [])
    if skewed:
        findings.append(f'<div class="key-finding">📊 Skewed distributions: {", ".join(skewed[:3])}</div>')
    
    return ''.join(findings) if findings else '<div class="key-finding">✅ Data looks clean!</div>'





def generate_eda(dataset_id: str, use_sample: bool = False, sample_size: int = 10000) -> Dict:
    """
    Comprehensive EDA with visualization support.
    
    Args:
        dataset_id: Dataset ID
        use_sample: Whether to use sample for wide datasets
        sample_size: Sample size for analysis
    """
    try:
        # Prefer clean data, fall back to ingested (normalized) data
        try:
            df = load_clean(dataset_id)
            data_source = "clean"
        except:
            from backend.services.utils import load_ingested
            df = load_ingested(dataset_id)
            data_source = "ingested"
        
        # Sample handling for wide datasets
        if use_sample and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            is_sampled = True
        else:
            is_sampled = False
        
        # Column selection for very wide datasets
        col_selection = select_columns_for_eda(df, max_cols=50)
        analyze_cols = col_selection.get("numeric", []) + col_selection.get("categorical", [])
        
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
        
        # ✅ GENERATE VISUALIZATIONS (BASE64)
        numeric_cols = _numeric_cols(df)
        visualizations = {
            "heatmap": generate_heatmap_base64(df),
            "boxplot": generate_boxplot_base64(df, numeric_cols[:20]) if numeric_cols else ""
        }
        
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "data_source": data_source,
            "is_sampled": is_sampled,
            "sample_size": sample_size if is_sampled else None,
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
            },
            "visualizations": visualizations,
            "column_selection": col_selection
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "EDA_FAILED"
        }


def generate_data_profile(dataset_id: str, use_sample: bool = False, sample_size: int = 10000) -> Dict:
    """
    Generate downloadable data profile/report with HTML export.
    """
    eda_result = generate_eda(dataset_id, use_sample=use_sample, sample_size=sample_size)
    
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
    
    # ✅ Generate HTML report
    html_report = generate_html_eda_report(eda_result, dataset_id)
    
    return {
        "status": "ok",
        "report": report,
        "download_format": "JSON + HTML",
        "data_for_export": eda_result,
        "html_report": html_report
    }
