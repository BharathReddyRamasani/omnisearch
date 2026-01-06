import os
import io
import json
import uuid
import pandas as pd
from fastapi import UploadFile, HTTPException
from charset_normalizer import from_bytes
from backend.services.utils import raw_path, datasetdir

# ---------------------------
# Helpers
# ---------------------------

def detect_encoding(raw: bytes) -> str:
    result = from_bytes(raw[:10000]).best()
    return result.encoding if result else "utf-8"

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    seen = {}

    for col in df.columns.astype(str):
        c = col.strip().lower()
        c = "".join(ch if ch.isalnum() else "_" for ch in c)
        c = "_".join(filter(None, c.split("_")))

        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0

        new_cols.append(c)

    df.columns = new_cols
    return df

# ---------------------------
# Main upload logic
# ---------------------------

async def process_upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    dataset_id = str(uuid.uuid4())[:8]
    raw_p = raw_path(dataset_id)

    # Read file bytes
    raw = await file.read()

    try:
        if file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            enc = detect_encoding(raw)
            df = pd.read_csv(io.StringIO(raw.decode(enc, errors="replace")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Normalize & basic cleaning
    df = normalize_columns(df)
    df = df.dropna(how="all")

    # Save raw
    df.to_csv(raw_p, index=False)

    # Also save to dataset dir for consistency
    dpath = datasetdir(dataset_id)
    df.to_csv(os.path.join(dpath, "raw.csv"), index=False)

    # Save schema
    schema = {c: str(df[c].dtype) for c in df.columns}
    with open(os.path.join(dpath, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records")
    }
