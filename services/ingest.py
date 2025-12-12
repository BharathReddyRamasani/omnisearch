# services/ingest.py
from charset_normalizer import from_bytes
import io
import pandas as pd
import re

def detect_encoding(sample_bytes: bytes):
    res = from_bytes(sample_bytes).best()
    return res.encoding if res else "utf-8"

def clean_col(name):
    n = str(name).strip().lower()
    # replace non-alphanumeric with underscore
    n = re.sub(r"[^\w]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n or "col"

def read_csv_sample_bytes(raw_bytes: bytes, nrows=2000):
    enc = detect_encoding(raw_bytes[:20000])
    text = raw_bytes.decode(enc, errors="replace")
    # sample read
    df = pd.read_csv(io.StringIO(text), nrows=nrows)
    original_cols = list(df.columns.astype(str))
    new_cols = [clean_col(c) for c in original_cols]
    df.columns = new_cols
    mapping = dict(zip(original_cols, new_cols))
    return df, mapping, enc
