
# backend/app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.ingest import read_csv_sample_bytes, clean_col
import pandas as pd
import io
from services.eda import generate_eda



app = FastAPI(title="OmniSearch - Dev API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()

        # CASE 1: CSV
        if filename.endswith(".csv"):
            df, mapping, encoding = read_csv_sample_bytes(content, nrows=1000)

        # CASE 2: Excel
        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            import pandas as pd, io
            excel_df = pd.read_excel(io.BytesIO(content), nrows=1000)

            original_cols = [str(c) for c in excel_df.columns]
            new_cols = [clean_col(c) for c in original_cols]

            excel_df.columns = new_cols
            mapping = dict(zip(original_cols, new_cols))

            df = excel_df
            encoding = "binary-excel"

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        preview = df.head(5).to_dict(orient="records")
        cols = [{"original": o, "normalized": n} for o, n in mapping.items()]

        return {
            "filename": file.filename,
            "encoding": encoding,
            "columns": cols,
            "preview": preview,
            "stats": {
                "rows_sampled": len(df),
                "columns": len(df.columns)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")

@app.post("/profile")
async def profile(data: dict):
    try:
        rows = data.get("rows", [])
        if not rows:
            raise ValueError("No rows received for EDA")

        df = pd.DataFrame(rows)

        # Convert numeric-looking strings to numbers where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        from services.eda import generate_eda
        eda = generate_eda(df)

        return eda

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to generate EDA: {str(e)}")
