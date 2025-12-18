import pandas as pd

def detect_quality_issues(df: pd.DataFrame) -> dict:
    issues = {}

    for col in df.columns:
        col_issues = {}

        missing_pct = df[col].isna().mean() * 100
        if missing_pct > 0:
            col_issues["missing_pct"] = round(missing_pct, 2)

        if df[col].dtype == "object":
            try:
                pd.to_numeric(df[col])
            except:
                col_issues["mixed_or_non_numeric"] = True

        if df[col].nunique() <= 1:
            col_issues["zero_variance"] = True

        if col_issues:
            issues[col] = col_issues

    return issues
