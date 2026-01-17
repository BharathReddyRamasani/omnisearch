from curses import raw
import os
import io
import json
import uuid
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, List, Any
from fastapi import UploadFile, HTTPException
from charset_normalizer import from_bytes
from dateutil import parser as dateutil_parser
from backend.services.utils import raw_path, datasetdir

# =====================================================
# CONFIGURATION & LOGGING
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Size limits (user-friendly)
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB
MAX_COLUMNS = 500
MAX_ROWS_SAMPLE = 100000  # Max rows to sample for inference

# Encoding detection thresholds
ENCODING_CONFIDENCE_THRESHOLD = 0.7  # Only accept encodings with >= 70% confidence
ENCODING_DETECTION_SAMPLE_SIZE = 50000  # Bytes to sample for detection

# =====================================================
# ENCODING DETECTION (Industrial-Grade)
# =====================================================

def detect_encoding(raw: bytes) -> Tuple[str, float, str]:
    """
    Industrial-grade encoding detection using charset-normalizer.
    
    Returns:
        Tuple[encoding_name, confidence, detection_method]
    """
    if not raw:
        return "utf-8", 1.0, "empty_file"
    
    # Sample data for detection
    sample = raw[:min(ENCODING_DETECTION_SAMPLE_SIZE, len(raw))]
    
    try:
        # Use charset-normalizer for detection
        detected = from_bytes(sample).best()
        
        if detected is None:
            logger.warning("charset-normalizer returned None, falling back to utf-8")
            return "utf-8", 0.0, "fallback_none"
        
        encoding = detected.encoding
        confidence = float(detected.confidence)
        
        logger.info(f"Detected encoding: {encoding} with confidence: {confidence:.2%}")
        
        # Check confidence threshold
        if confidence >= ENCODING_CONFIDENCE_THRESHOLD:
            return encoding, confidence, "charset_normalizer"
        
        # Low confidence fallback strategy
        logger.warning(f"Encoding confidence {confidence:.2%} below threshold {ENCODING_CONFIDENCE_THRESHOLD:.0%}")
        
        # Try common fallbacks for different regions
        fallback_candidates = [
            ("utf-8", 0.95),
            ("iso-8859-1", 0.80),
            ("cp1252", 0.75),  # Windows-1252 (Excel exports)
            ("utf-16", 0.70),
        ]
        
        for candidate_enc, candidate_conf in fallback_candidates:
            try:
                # Test if candidate can decode without errors
                raw.decode(candidate_enc, errors="strict")
                logger.info(f"Fallback to {candidate_enc} (confidence: {candidate_conf:.0%})")
                return candidate_enc, candidate_conf, "fallback_tested"
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Final fallback
        logger.warning("All fallback strategies failed, using utf-8 with error replacement")
        return "utf-8", 0.5, "fallback_final"
        
    except Exception as e:
        logger.error(f"Encoding detection error: {str(e)}, falling back to utf-8")
        return "utf-8", 0.0, f"error_{type(e).__name__}"


# =====================================================
# COLUMN NORMALIZATION
# =====================================================

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Normalize column names to clean, consistent format.
    
    Returns:
        Tuple[normalized_df, mapping_dict]
        mapping_dict: {original_col_name: normalized_col_name}
    """
    df = df.copy()
    mapping = {}
    new_cols = []
    seen = {}

    for col in df.columns.astype(str):
        original = col
        
        # Step 1: Strip whitespace and convert to lowercase
        c = col.strip().lower()
        
        # Step 2: Replace non-alphanumeric with underscore
        c = "".join(ch if ch.isalnum() else "_" for ch in c)
        
        # Step 3: Deduplicate underscores
        c = "_".join(filter(None, c.split("_")))
        
        # Step 4: Handle duplicates
        if c in seen:
            seen[c] += 1
            c = f"{c}_{seen[c]}"
        else:
            seen[c] = 0
        
        new_cols.append(c)
        mapping[original] = c
        
        logger.info(f"Normalized column: '{original}' → '{c}'")

    df.columns = new_cols
    return df, mapping


# =====================================================
# TYPE INFERENCE & COERCION
# =====================================================

def infer_and_cast(df: pd.DataFrame, sample_rows: int = 1000) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Infer column types and apply smart coercion.
    
    Returns:
        Tuple[casted_df, coercion_report]
        coercion_report: {
            'col_name': {
                'inferred_type': str,
                'coercion_count': int,
                'coercion_method': str,
                'nulls_created': int
            }
        }
    """
    df = df.copy()
    coercion_report = {}
    
    # Sample for inference to avoid memory issues on huge datasets
    sample_size = min(sample_rows, len(df))
    sample_df = df.head(sample_size)
    
    for col in df.columns:
        col_report = {
            'original_dtype': str(df[col].dtype),
            'inferred_type': 'object',
            'coercion_count': 0,
            'coercion_method': 'none',
            'nulls_created': 0
        }
        
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            col_report['inferred_type'] = str(df[col].dtype)
            coercion_report[col] = col_report
            continue
        
        # Try numeric coercion
        try:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            null_count = numeric_vals.isna().sum()
            original_null_count = df[col].isna().sum()
            new_nulls = null_count - original_null_count
            
            if new_nulls < len(df) * 0.3:  # Allow up to 30% coercion
                df[col] = numeric_vals
                col_report['inferred_type'] = 'numeric'
                col_report['coercion_count'] = new_nulls
                col_report['coercion_method'] = 'numeric'
                col_report['nulls_created'] = new_nulls
                logger.info(f"Coerced '{col}' to numeric ({new_nulls} nulls created)")
                coercion_report[col] = col_report
                continue
        except Exception as e:
            logger.debug(f"Numeric coercion failed for '{col}': {str(e)}")
        
        # Try date parsing
        try:
            # Only try on small non-numeric columns
            if sample_df[col].dtype == 'object' and sample_size > 0:
                sample_vals = sample_df[col].dropna().head(50)
                if len(sample_vals) > 0:
                    # Check if values look like dates
                    date_attempts = 0
                    for val in sample_vals:
                        try:
                            dateutil_parser.parse(str(val), fuzzy=False)
                            date_attempts += 1
                        except (ValueError, TypeError):
                            pass
                    
                    if date_attempts / len(sample_vals) > 0.8:  # 80% success rate
                        date_vals = pd.to_datetime(df[col], errors="coerce")
                        null_count = date_vals.isna().sum()
                        original_null_count = df[col].isna().sum()
                        new_nulls = null_count - original_null_count
                        
                        if new_nulls < len(df) * 0.3:  # Allow up to 30% coercion
                            df[col] = date_vals
                            col_report['inferred_type'] = 'datetime'
                            col_report['coercion_count'] = new_nulls
                            col_report['coercion_method'] = 'date_parsing'
                            col_report['nulls_created'] = new_nulls
                            logger.info(f"Coerced '{col}' to datetime ({new_nulls} nulls created)")
                            coercion_report[col] = col_report
                            continue
        except Exception as e:
            logger.debug(f"Date coercion failed for '{col}': {str(e)}")
        
        # Default: keep as object
        col_report['inferred_type'] = 'object'
        coercion_report[col] = col_report
    
    return df, coercion_report


# =====================================================
# VALIDATION & CAPS
# =====================================================

def validate_file_constraints(raw_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validate file size, dimensions, and other constraints.
    
    Returns:
        Dict with validation result and friendly error messages
    """
    # File size check
    file_size = len(raw_bytes)
    if file_size > MAX_FILE_SIZE_BYTES:
        return {
            "valid": False,
            "error": f"File too large: {file_size / 1024 / 1024:.1f}MB exceeds maximum {MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB",
            "error_code": "FILE_SIZE_EXCEEDED"
        }
    
    if file_size == 0:
        return {
            "valid": False,
            "error": "Uploaded file is empty",
            "error_code": "EMPTY_FILE"
        }
    
    logger.info(f"File size validation passed: {file_size / 1024:.1f}KB")
    return {"valid": True}


def validate_dataframe_constraints(df: pd.DataFrame, dataset_id: str) -> Dict[str, Any]:
    """
    Validate DataFrame dimensions against limits.
    
    Returns:
        Dict with validation result and friendly error messages
    """
    rows = len(df)
    cols = len(df.columns)
    
    # Row check
    if rows > MAX_ROWS_SAMPLE:
        return {
            "valid": False,
            "error": f"Dataset too large: {rows:,} rows exceeds maximum {MAX_ROWS_SAMPLE:,} rows",
            "error_code": "ROWS_EXCEEDED"
        }
    
    # Column check
    if cols > MAX_COLUMNS:
        return {
            "valid": False,
            "error": f"Too many columns: {cols} exceeds maximum {MAX_COLUMNS}",
            "error_code": "COLUMNS_EXCEEDED"
        }
    
    logger.info(f"DataFrame validation passed: {rows:,} rows × {cols} columns")
    return {"valid": True}


# =====================================================
# MAIN UPLOAD LOGIC
# =====================================================

async def process_upload(file: UploadFile) -> Dict[str, Any]:
    """
    Complete upload pipeline with industrial-grade processing.
    
    Returns:
        Dict with upload status, dataset_id, columns, mapping, encoding info, and preview
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    dataset_id = str(uuid.uuid4())[:8]
    
    logger.info(f"Processing upload: {file.filename} → dataset_id={dataset_id}")
    
    # Read file bytes
    raw = await file.read()
    
    # Validate file constraints
    validation = validate_file_constraints(raw, file.filename)
    if not validation["valid"]:
        logger.error(f"File validation failed: {validation['error']}")
        raise HTTPException(
            status_code=400, 
            detail=validation["error"]
        )
    
    # Detect encoding
    encoding, confidence, detection_method = detect_encoding(raw)
    logger.info(f"Encoding: {encoding} (confidence: {confidence:.2%}, method: {detection_method})")
    
    # Read file with detected encoding (SAMPLE-BASED to prevent memory blowup)
    try:
        if file.filename.endswith((".xlsx", ".xls")):
            logger.info("Reading as Excel file (sample-based)")
            df = pd.read_excel(
            io.BytesIO(raw),
            nrows=MAX_ROWS_SAMPLE
            )

        else:
            logger.info(
                f"Reading CSV with encoding={encoding}, "
                f"sample_limit={MAX_ROWS_SAMPLE}, robust parsing enabled"
            )

            df = pd.read_csv(
                io.BytesIO(raw),              # ⬅ avoid full decode to string
                encoding=encoding,
                nrows=MAX_ROWS_SAMPLE,        # ⬅ hard safety cap
                sep=None,                     # ⬅ delimiter auto-detection
                engine="python",              # ⬅ required for sep=None
                on_bad_lines="skip",          # ⬅ dirty rows tolerated
                low_memory=False              # ⬅ consistent dtype inference
            )

    except Exception as e:
        logger.error("Failed to read uploaded file", exc_info=e)
        raise HTTPException(
            status_code=400,
            detail=(
                "Failed to parse the uploaded file. "
                "Ensure it is a valid CSV/Excel file with consistent structure."
            )
        )
        
    # Validate DataFrame constraints
    validation = validate_dataframe_constraints(df, dataset_id)
    if not validation["valid"]:
        logger.error(f"DataFrame validation failed: {validation['error']}")
        raise HTTPException(
            status_code=400, 
            detail=validation["error"]
        )
    
    if df.empty:
        logger.error("DataFrame is empty after reading")
        raise HTTPException(status_code=400, detail="Uploaded file contains no data")
    
    # Normalize columns (get mapping)
    df, col_mapping = normalize_columns(df)
    
    # Type inference & coercion
    df, coercion_report = infer_and_cast(df)
    
    # Basic cleaning
    df = df.dropna(how="all")
    
    # Save TRUE raw file (unaltered)
    raw_p = raw_path(dataset_id)
    with open(raw_p, "wb") as f:
        f.write(raw)

    logger.info(f"Saved raw CSV to {raw_p}")
    
    # Save to dataset dir
    dpath = datasetdir(dataset_id)
    # Save ingested (normalized + typed)
    ingested_path = os.path.join(dpath, "ingested.csv")
    df.to_csv(ingested_path, index=False)

    
    # Save schema
    schema = {c: str(df[c].dtype) for c in df.columns}
    with open(os.path.join(dpath, "schema.json"), "w") as f:
        json.dump(schema, f, indent=2)
    
    # Save upload metadata
    # Save upload metadata (DATA GOVERNANCE SAFE)
    upload_metadata = {
        "dataset_id": dataset_id,
        "original_filename": file.filename,
        "upload_timestamp": datetime.utcnow().isoformat(),
    
        "encoding": {
            "detected": encoding,
            "confidence": confidence,
            "detection_method": detection_method
        },
    
        "dimensions": {
            "rows": len(df),
            "columns": len(df.columns),
            "sample_limit": MAX_ROWS_SAMPLE,
            "is_sampled": len(df) >= MAX_ROWS_SAMPLE
        },
    
        "file_size_bytes": len(raw),
    
        # 🔒 ARTIFACT LINEAGE (CRITICAL)
        "artifacts": {
            "source_raw": raw_p,          # exact uploaded bytes (immutable)
            "ingested": ingested_path     # normalized + type-inferred data
        },
    
        "column_mapping": col_mapping,
        "type_coercion": coercion_report,
    
        # Governance control
        "column_mapping_confirmed": False
    }

    with open(os.path.join(dpath, "upload_metadata.json"), "w") as f:
        json.dump(upload_metadata, f, indent=2)
    
    logger.info(f"Upload complete for {dataset_id}: {len(df)} rows × {len(df.columns)} columns")
    
    return {
        "status": "ok",
        "dataset_id": dataset_id,
        "rows": len(df),
        "columns": list(df.columns),
        "is_sampled": len(df) >= MAX_ROWS_SAMPLE,
        "sample_limit": MAX_ROWS_SAMPLE,
        "column_mapping": col_mapping,  # NEW: Original → Normalized mapping
        "encoding": {
            "detected": encoding,
            "confidence": confidence,
            "detection_method": detection_method
        },
        "coercion_summary": {
            col: {
                "type": report.get("inferred_type"),
                "coercions": report.get("coercion_count", 0),
                "method": report.get("coercion_method", "none")
            }
            for col, report in coercion_report.items()
        },
        "preview": df.head(5).to_dict(orient="records")
    }
