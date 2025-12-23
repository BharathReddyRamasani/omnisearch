import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from backend.services.utils import load_df, clean_dataframe

def generate_eda(dataset_id: str):
    try:
        df = clean_dataframe(load_df(dataset_id))
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        before_summary = {}
        after_summary = {}
        plots_before = {}
        plots_after = {}
        outliers = {}
        
        for col in num_cols[:3]:  # Top 3 numeric columns
            s = df[col].dropna()
            if s.empty or len(s) < 4:
                continue
                
            before_summary[col] = s.describe().to_dict()
            
            # Calculate IQR bounds EXPLICITLY
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            outlier_lower_bound = q1 - 1.5 * iqr  # ✅ EXPLICIT NAMES
            outlier_upper_bound = q3 + 1.5 * iqr  # ✅ EXPLICIT NAMES
            
            # Count outliers
            outlier_mask = (s < outlier_lower_bound) | (s > outlier_upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[col] = {
                "count": int(outlier_count),
                "lower": float(outlier_lower_bound),
                "upper": float(outlier_upper_bound)
            }
            
            # Before plot
            fig, ax = plt.subplots(figsize=(6, 4))
            bins = min(30, max(5, len(s) // 5))
            s.hist(ax=ax, bins=bins)
            ax.set_title(f'Before Cleaning: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            plots_before[col] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
            
            # After clipping outliers
            clipped = s.clip(lower=outlier_lower_bound, upper=outlier_upper_bound)
            after_summary[col] = clipped.describe().to_dict()
            
            fig, ax = plt.subplots(figsize=(6, 4))
            clipped.hist(ax=ax, bins=bins)
            ax.set_title(f'After Outlier Clipping: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close(fig)
            plots_after[col] = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        
        return {
            "status": "ok",
            "eda": {
                "rows": len(df),
                "columns": len(df.columns),
                "missing": df.isnull().sum().to_dict(),
                "dtypes": {c: str(df[c].dtype) for c in df.columns},
                "before_summary": before_summary,
                "outliers": outliers,
                "plots_before": plots_before,
                "after_summary": after_summary,
                "plots_after": plots_after
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "eda": {}
        }
