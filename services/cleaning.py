import pandas as pd

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()

    # strip strings
    for col in clean_df.select_dtypes(include="object").columns:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    # numeric coercion
    for col in clean_df.columns:
        try:
            clean_df[col] = pd.to_numeric(clean_df[col])
        except:
            pass

    # drop duplicates
    clean_df = clean_df.drop_duplicates()

    return clean_df
