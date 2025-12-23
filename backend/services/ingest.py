import os
import json
import pandas as pd
from fastapi import UploadFile, HTTPException
from backend.services.utils import dataset_dir, clean_dataframe, load_df, validate_target

async def process_upload(file: UploadFile):
    dataset_id = file.filename.replace('.', '_')[:20]
    dpath = dataset_dir(dataset_id)
    
    if file.filename.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file.file)
    else:
        df = pd.read_csv(file.file)
    
    df = clean_dataframe(df)
    df.to_csv(os.path.join(dpath, "raw.csv"), index=False)
    
    schema = {c: str(df[c].dtype) for c in df.columns}
    json.dump(schema, open(os.path.join(dpath, "schema.json"), 'w'), indent=2)
    
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient='records')
    }

def get_schema_logic(dataset_id: str):
    df = load_df(dataset_id)
    schema = {}
    target_analysis = {}
    
    for col in df.columns:
        schema[col] = str(df[col].dtype)
        reason = validate_target(df[col])
        target_analysis[col] = {
            "valid": reason is None,
            "reasons": [reason] if reason else []
        }
    
    return {
        "status": "ok",
        "schema": schema,
        "target_analysis": target_analysis
    }

def get_meta_logic(dataset_id: str):
    path = os.path.join(dataset_dir(dataset_id), "meta.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Train a model first")
    return json.load(open(path))
