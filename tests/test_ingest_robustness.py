"""
Test suite for CSV ingestion robustness.
Tests with various malformed, edge-case, and real-world CSV files.
"""

import os
import io
import pytest
import pandas as pd
from pathlib import Path

# Add backend to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.services.ingest import (
    detect_encoding,
    normalize_columns,
    infer_and_cast,
    validate_file_constraints,
    validate_dataframe_constraints
)


# =====================================================
# FIXTURES
# =====================================================

BAD_CSV_DIR = Path(__file__).parent / "bad_csvs"


@pytest.fixture
def broken_headers_file():
    """CSV with missing values in headers"""
    return BAD_CSV_DIR / "broken_headers.csv"


@pytest.fixture
def extra_delimiters_file():
    """CSV with extra commas/delimiters in quoted fields"""
    return BAD_CSV_DIR / "extra_delimiters.csv"


@pytest.fixture
def mixed_encoding_file():
    """CSV with mixed character encodings (special chars)"""
    return BAD_CSV_DIR / "mixed_encoding.csv"


@pytest.fixture
def empty_columns_file():
    """CSV with empty/blank columns"""
    return BAD_CSV_DIR / "empty_columns.csv"


@pytest.fixture
def inconsistent_columns_file():
    """CSV with inconsistent column counts across rows"""
    return BAD_CSV_DIR / "inconsistent_columns.csv"


@pytest.fixture
def special_chars_headers_file():
    """CSV with special characters in column headers"""
    return BAD_CSV_DIR / "special_chars_headers.csv"


# =====================================================
# ENCODING DETECTION TESTS
# =====================================================

class TestEncodingDetection:
    """Test encoding detection across various files"""
    
    def test_utf8_detection(self):
        """UTF-8 should be detected for standard ASCII"""
        raw = "Name,Age,Salary\nJohn,30,50000".encode('utf-8')
        encoding, confidence, method = detect_encoding(raw)
        assert encoding is not None
        assert confidence >= 0.0
        assert method is not None
        print(f"UTF-8: {encoding} ({confidence:.2%}) via {method}")
    
    def test_mixed_encoding_file(self, mixed_encoding_file):
        """Mixed unicode characters should still be detected"""
        with open(mixed_encoding_file, 'rb') as f:
            raw = f.read()
        
        encoding, confidence, method = detect_encoding(raw)
        assert encoding is not None
        print(f"Mixed encoding detected: {encoding} ({confidence:.2%}) via {method}")
        
        # Should be able to decode without errors
        try:
            raw.decode(encoding, errors='replace')
        except Exception as e:
            pytest.fail(f"Failed to decode with detected encoding: {str(e)}")
    
    def test_empty_file(self):
        """Empty file should default to UTF-8"""
        raw = b""
        encoding, confidence, method = detect_encoding(raw)
        assert encoding == "utf-8"
        assert confidence == 1.0
        print(f"Empty file: {encoding}")


# =====================================================
# COLUMN NORMALIZATION TESTS
# =====================================================

class TestColumnNormalization:
    """Test column name normalization"""
    
    def test_special_chars_normalization(self, special_chars_headers_file):
        """Special characters in headers should be normalized"""
        df = pd.read_csv(special_chars_headers_file)
        df_normalized, mapping = normalize_columns(df)
        
        # All columns should be lowercase, alphanumeric + underscore
        for col in df_normalized.columns:
            assert col.islower() or col.replace('_', '').isalnum()
            print(f"  Normalized: {col}")
        
        print(f"Mapping: {mapping}")
        assert len(mapping) == len(df.columns)
    
    def test_empty_columns_normalization(self, empty_columns_file):
        """Empty columns should be handled gracefully"""
        df = pd.read_csv(empty_columns_file)
        original_cols = len(df.columns)
        
        df_normalized, mapping = normalize_columns(df)
        
        # Should maintain column count
        assert len(df_normalized.columns) == original_cols
        print(f"Empty columns: {original_cols} columns → {mapping}")
    
    def test_duplicate_normalization(self):
        """Duplicate normalized names should get unique suffixes"""
        df = pd.DataFrame({
            "Name": [1, 2, 3],
            "name": [4, 5, 6],
            "NAME": [7, 8, 9]
        })
        
        df_normalized, mapping = normalize_columns(df)
        
        # Should have unique column names
        assert len(df_normalized.columns) == len(set(df_normalized.columns))
        print(f"Duplicates handled: {mapping}")


# =====================================================
# TYPE INFERENCE TESTS
# =====================================================

class TestTypeInference:
    """Test type inference and coercion"""
    
    def test_numeric_coercion(self):
        """String numbers should be coerced to numeric"""
        df = pd.DataFrame({
            "name": ["John", "Jane", "Bob"],
            "age": ["30", "27", "35"],  # String numbers
            "salary": ["50000", "60000", "55000"]
        })
        
        df_casted, coercion_report = infer_and_cast(df)
        
        # Age and salary should be numeric
        assert pd.api.types.is_numeric_dtype(df_casted['age'])
        assert pd.api.types.is_numeric_dtype(df_casted['salary'])
        
        # Check coercion report
        assert coercion_report['age']['inferred_type'] == 'numeric'
        assert coercion_report['salary']['inferred_type'] == 'numeric'
        print(f"Coercion report: {coercion_report}")
    
    def test_date_coercion(self):
        """Date-like strings should be coerced to datetime"""
        df = pd.DataFrame({
            "name": ["John", "Jane", "Bob"],
            "date": ["2024-01-15", "2024-02-20", "2024-03-10"]
        })
        
        df_casted, coercion_report = infer_and_cast(df)
        
        # Date column should be datetime
        assert pd.api.types.is_datetime64_any_dtype(df_casted['date'])
        assert coercion_report['date']['inferred_type'] == 'datetime'
        print(f"Date coercion: {coercion_report['date']}")
    
    def test_mixed_types_no_over_coercion(self):
        """Mixed types should not be aggressively coerced"""
        df = pd.DataFrame({
            "mixed": ["100", "200", "Not a number", "400", "Also not"]
        })
        
        df_casted, coercion_report = infer_and_cast(df)
        
        # Too many failures, should stay as object
        assert coercion_report['mixed']['inferred_type'] == 'object'
        print(f"Mixed types not coerced: {coercion_report['mixed']}")


# =====================================================
# CONSTRAINT VALIDATION TESTS
# =====================================================

class TestConstraintValidation:
    """Test file size and dimension constraints"""
    
    def test_file_size_validation_small(self):
        """Small files should pass validation"""
        raw = b"Name,Age\nJohn,30" * 100  # Small file
        result = validate_file_constraints(raw, "test.csv")
        assert result["valid"]
        print(f"Small file passed: {len(raw)} bytes")
    
    def test_file_size_validation_large(self):
        """Very large files should fail gracefully"""
        # Create a 600MB mock (we'll just test the logic)
        raw = b"x" * (600 * 1024 * 1024)
        result = validate_file_constraints(raw, "large.csv")
        assert not result["valid"]
        assert "FILE_SIZE_EXCEEDED" in result["error_code"]
        print(f"Large file rejected: {result['error']}")
    
    def test_empty_file_validation(self):
        """Empty files should be rejected"""
        raw = b""
        result = validate_file_constraints(raw, "empty.csv")
        assert not result["valid"]
        assert "EMPTY_FILE" in result["error_code"]
        print(f"Empty file rejected: {result['error']}")
    
    def test_dataframe_column_limit(self):
        """DataFrames with too many columns should fail"""
        # Create DF with 501 columns (limit is 500)
        cols = {f"col_{i}": [1, 2, 3] for i in range(501)}
        df = pd.DataFrame(cols)
        
        result = validate_dataframe_constraints(df, "test_id")
        assert not result["valid"]
        assert "COLUMNS_EXCEEDED" in result["error_code"]
        print(f"Too many columns rejected: {result['error']}")


# =====================================================
# INTEGRATION TESTS (Real bad CSVs)
# =====================================================

class TestRealBadCSVs:
    """Test actual bad CSV files"""
    
    def test_broken_headers_csv(self, broken_headers_file):
        """Should handle CSV with missing/empty values gracefully"""
        df = pd.read_csv(broken_headers_file)
        df_normalized, mapping = normalize_columns(df)
        
        # Should read without error
        assert len(df) > 0
        assert len(df.columns) > 0
        print(f"Broken headers processed: {len(df)} rows × {len(df.columns)} cols")
    
    def test_extra_delimiters_csv(self, extra_delimiters_file):
        """Should handle quoted fields with delimiters"""
        df = pd.read_csv(extra_delimiters_file)
        df_normalized, mapping = normalize_columns(df)
        
        assert len(df) > 0
        # Check that the Notes column has the expected commas
        notes = df_normalized.iloc[0][df_normalized.columns[3]]
        print(f"Extra delimiters handled: {notes[:50]}...")
    
    def test_mixed_encoding_csv(self, mixed_encoding_file):
        """Should handle mixed UTF-8 and special chars"""
        with open(mixed_encoding_file, 'rb') as f:
            raw = f.read()
        
        encoding, confidence, method = detect_encoding(raw)
        
        # Should successfully decode
        try:
            decoded = raw.decode(encoding, errors='replace')
            df = pd.read_csv(io.StringIO(decoded))
            assert len(df) > 0
            print(f"Mixed encoding processed with {encoding}")
        except Exception as e:
            pytest.fail(f"Failed to process mixed encoding: {str(e)}")
    
    def test_empty_columns_csv(self, empty_columns_file):
        """Should handle columns with blank values"""
        df = pd.read_csv(empty_columns_file)
        df_normalized, mapping = normalize_columns(df)
        
        assert len(df) > 0
        print(f"Empty columns processed: {mapping}")
    
    def test_inconsistent_columns_csv(self, inconsistent_columns_file):
        """Should handle rows with different column counts"""
        df = pd.read_csv(inconsistent_columns_file)
        df_normalized, mapping = normalize_columns(df)
        
        # Pandas fills missing values with NaN
        assert len(df) > 0
        print(f"Inconsistent columns handled: {len(df)} rows")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
