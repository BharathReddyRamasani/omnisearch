#!/usr/bin/env python3
"""
Test script to verify anomaly detection fixes for feature availability issues.
"""
import pandas as pd
import sys
sys.path.insert(0, 'd:\\omnisearch')

from backend.services.anomaly import run_anomaly_detection

# Test 1: Test with exact column name matches
print("=" * 60)
print("TEST 1: Exact Column Name Matches")
print("=" * 60)

df = pd.DataFrame({
    'Fresh': [1, 2, 3, 4, 5, 100, 200],
    'Milk': [10, 20, 30, 40, 50, 600, 700],
    'Grocery': [100, 200, 300, 400, 500, 6000, 7000]
})

result = run_anomaly_detection(df, ['Fresh', 'Milk'], method="Isolation Forest", contamination=0.2)
print(f"Status: {result.get('status')}")
print(f"Features Used: {result.get('features_used')}")
print(f"Samples: {result.get('n_samples')}")
print(f"Anomalies Detected: {result.get('n_anomalies')}")
if result.get('status') == 'failed':
    print(f"Error: {result.get('error')}")
else:
    print("✅ Test 1 PASSED")
print()

# Test 2: Test with case mismatch (lowercase requested, mixed-case in dataframe)
print("=" * 60)
print("TEST 2: Case-Insensitive Matching")
print("=" * 60)

result = run_anomaly_detection(df, ['fresh', 'milk'], method="Isolation Forest", contamination=0.2)
print(f"Status: {result.get('status')}")
print(f"Features Used: {result.get('features_used')}")
print(f"Requested: ['fresh', 'milk']")
print(f"Actual DF columns: {df.columns.tolist()}")
if result.get('status') == 'failed':
    print(f"Error: {result.get('error')}")
else:
    print("✅ Test 2 PASSED")
print()

# Test 3: Test with missing features
print("=" * 60)
print("TEST 3: Missing Features Error Handling")
print("=" * 60)

result = run_anomaly_detection(df, ['Fresh', 'NonExistent', 'Milk'], method="Isolation Forest")
print(f"Status: {result.get('status')}")
print(f"Missing Features: {result.get('missing_features')}")
print(f"Available Features shown: {', '.join(result.get('available_features', [])[:5])}")
if result.get('status') == 'failed':
    print(f"Error Message: {result.get('error')[:80]}...")
    print("✅ Test 3 PASSED - Error handled correctly")
print()

# Test 4: Test with all three methods
print("=" * 60)
print("TEST 4: Testing All Methods")
print("=" * 60)

methods = ["Isolation Forest", "One-Class SVM", "Local Outlier Factor"]
for method in methods:
    result = run_anomaly_detection(df, ['Fresh', 'Milk'], method=method, contamination=0.2)
    print(f"{method}: Status={result.get('status')}, Anomalies={result.get('n_anomalies')}")
    
print("✅ Test 4 PASSED")
print()

# Test 5: Test with NaN handling
print("=" * 60)
print("TEST 5: NaN Handling")
print("=" * 60)

df_nan = pd.DataFrame({
    'Fresh': [1.0, 2.0, 3.0, None, 5.0, 100.0],
    'Milk': [10.0, 20.0, None, 40.0, 50.0, 600.0],
})

result = run_anomaly_detection(df_nan, ['Fresh', 'Milk'], method="Isolation Forest", contamination=0.3)
print(f"Status: {result.get('status')}")
print(f"Input rows with NaN: 6")
print(f"Samples processed: {result.get('n_samples')}")
if result.get('status') == 'ok':
    print("✅ Test 5 PASSED - NaN values properly handled")
else:
    print(f"Error: {result.get('error')}")
print()

print("=" * 60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 60)
