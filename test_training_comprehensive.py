#!/usr/bin/env python
"""
Comprehensive test to verify training works with column name mapping fix across multiple datasets.
"""
import sys
import json
import os
import glob

sys.path.insert(0, '/d/omnisearch')

from backend.services.training import train_model_logic
from backend.services.utils import datasetdir

def test_dataset(dataset_id: str, target_col: str) -> bool:
    """Test training on a specific dataset"""
    try:
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_id}")
        print(f"Target: {target_col}")
        print(f"{'='*60}")
        
        result = train_model_logic(
            dataset_id=dataset_id,
            target=target_col,
            task=None,
            test_size=0.2,
            random_state=42,
            time_limit_seconds=300
        )
        
        status = result.get('status')
        if status == 'ok':
            print(f"✅ SUCCESS - Task: {result.get('task')}")
            return True
        else:
            print(f"❌ FAILED - Error: {result.get('error')}")
            return False
    except Exception as e:
        print(f"❌ EXCEPTION - {str(e)}")
        return False

def find_datasets_with_clean_data():
    """Find all datasets with clean.csv and metadata"""
    datasets = []
    datasets_dir = "data/datasets"
    
    for dataset_path in sorted(glob.glob(os.path.join(datasets_dir, "*")))[:3]:  # Test first 3
        if os.path.isdir(dataset_path):
            dataset_id = os.path.basename(dataset_path)
            clean_path = os.path.join(dataset_path, "clean.csv")
            meta_path = os.path.join(dataset_path, "upload_metadata.json")
            
            if os.path.exists(clean_path) and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                
                # Find a suitable target column (prefer those at the end)
                col_mapping = meta.get("column_mapping", {})
                if col_mapping:
                    # Get the last mapped column as potential target
                    target = list(col_mapping.values())[-1]
                    datasets.append((dataset_id, target))
    
    return datasets

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE TRAINING TEST SUITE")
    print("Testing column name mapping fix across datasets")
    print("=" * 60)
    
    # Find datasets with clean data
    datasets = find_datasets_with_clean_data()
    
    if not datasets:
        print("No datasets with clean data found!")
        # Fall back to manual test
        datasets = [("00f53e8a", "loan_status")]
    
    results = []
    for dataset_id, target_col in datasets:
        success = test_dataset(dataset_id, target_col)
        results.append((dataset_id, target_col, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, _, success in results if success)
    total = len(results)
    
    for dataset_id, target_col, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {dataset_id} (target: {target_col})")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} tests failed!")
        sys.exit(1)
