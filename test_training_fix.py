#!/usr/bin/env python
"""
Test script to verify training works with column name mapping fix.
"""
import sys
import json
import os

# Add backend to path
sys.path.insert(0, '/d/omnisearch')

from backend.services.training import train_model_logic
from backend.services.utils import datasetdir

def test_training_with_dataset(dataset_id: str, target_col: str):
    """Test training on a specific dataset with a target column"""
    print(f"\n{'='*60}")
    print(f"Testing dataset: {dataset_id}")
    print(f"Target column: {target_col}")
    print(f"{'='*60}")
    
    # Load upload metadata to see the column mapping
    meta_path = os.path.join(datasetdir(dataset_id), "upload_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            upload_meta = json.load(f)
        
        print(f"\nAvailable columns in dataset:")
        for orig, mapped in upload_meta.get("column_mapping", {}).items():
            print(f"  {orig:30} → {mapped}")
    
    # Run training
    print(f"\nStarting training...")
    result = train_model_logic(
        dataset_id=dataset_id,
        target=target_col,
        task=None,  # Auto-detect
        test_size=0.2,
        random_state=42,
        time_limit_seconds=300
    )
    
    # Display results
    print(f"\nResult status: {result.get('status')}")
    if result.get('status') == 'failed':
        print(f"Error: {result.get('error')}")
        print(f"Details: {result.get('details', 'None')}")
        if 'available_columns' in result:
            print(f"Available columns: {result.get('available_columns')}")
    else:
        print(f"✅ Training successful!")
        print(f"  Status: {result.get('status')}")
        print(f"  Message: {result.get('message', 'N/A')}")
        if 'task' in result:
            print(f"  Task: {result.get('task')}")
        # Check if result is nested under 'result' key
        inner_result = result.get('result', {}) if isinstance(result.get('result'), dict) else {}
        if inner_result:
            print(f"  Task: {inner_result.get('task')}")
            print(f"  Samples: {inner_result.get('n_samples')}")
            print(f"  Features: {inner_result.get('n_features')}")
    
    return result.get('status') != 'failed'

if __name__ == "__main__":
    # Test with dataset 00f53e8a (Loan Status dataset)
    # Based on the metadata, target column should be 'loan_status' (normalized)
    success = test_training_with_dataset("00f53e8a", "loan_status")
    
    if success:
        print(f"\n✅ Test PASSED!")
    else:
        print(f"\n❌ Test FAILED!")
        sys.exit(1)
