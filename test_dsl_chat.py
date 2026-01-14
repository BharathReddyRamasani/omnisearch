#!/usr/bin/env python3
"""
Test script for OmniSearch AI DSL-based chat system.
Tests the complete pipeline from natural language to JSON DSL execution.
"""

import sys
import os
import json

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.services.chat import DatasetContext, IntentDetector, DSLBuilder, DSLExecutor, get_chat_response

def test_intent_detection():
    """Test intent detection for all 7 action types"""
    print("\n" + "="*60)
    print("TEST 1: Intent Detection")
    print("="*60)
    
    detector = IntentDetector()
    
    test_cases = [
        ("Tell me about my data", "describe"),
        ("What's the average age?", "aggregate"),
        ("Show me data grouped by gender", "groupby"),
        ("Are there correlations between variables?", "correlation"),
        ("Show me the distribution of age", "distribution"),
        ("What's my model's accuracy?", "model_info"),
    ]
    
    for query, expected_action in test_cases:
        detected = detector.detect_intent(query)
        status = "✅" if detected == expected_action else "❌"
        print(f"{status} '{query}' → {detected} (expected: {expected_action})")

def test_dataset_context():
    """Test dataset context loading and validation"""
    print("\n" + "="*60)
    print("TEST 2: Dataset Context Loading")
    print("="*60)
    
    try:
        # Use first available dataset
        dataset_id = "018d7cde"
        context = DatasetContext(dataset_id)
        
        print(f"✅ Loaded dataset: {dataset_id}")
        print(f"   Shape: {context.df.shape[0]} rows × {len(context.columns)} columns")
        print(f"   Numeric columns: {context.numeric_cols}")
        print(f"   Categorical columns: {context.categorical_cols}")
        print(f"   Model available: {bool(context.model_meta)}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")

def test_dsl_builder():
    """Test DSL builder for all action types"""
    print("\n" + "="*60)
    print("TEST 3: DSL Builder")
    print("="*60)
    
    try:
        dataset_id = "018d7cde"
        context = DatasetContext(dataset_id)
        builder = DSLBuilder(context)
        
        test_queries = [
            "Describe the dataset",
            "What's the average and sum?",
            "Group by gender and show average salary",
            "Are there correlations?",
            "Show age distribution",
            "What about my model?",
        ]
        
        for query in test_queries:
            dsl = builder.build(query)
            action = dsl.get('action', 'unknown')
            print(f"\n  Query: '{query}'")
            print(f"  DSL: {json.dumps(dsl, indent=4)}")
    
    except Exception as e:
        print(f"❌ Error building DSL: {e}")

def test_dsl_executor():
    """Test DSL executor for data operations"""
    print("\n" + "="*60)
    print("TEST 4: DSL Executor")
    print("="*60)
    
    try:
        dataset_id = "018d7cde"
        context = DatasetContext(dataset_id)
        executor = DSLExecutor(context)
        
        # Test describe
        print("\n  Testing: describe")
        result = executor.execute_describe()
        if result and 'error' not in result:
            print(f"  ✅ Describe executed successfully")
            print(f"     Columns: {result.get('num_columns')}")
            print(f"     Rows: {result.get('num_rows')}")
        
        # Test aggregate
        print("\n  Testing: aggregate")
        result = executor.execute_aggregate(['Age'], {'Age': 'mean'})
        if result and 'error' not in result:
            print(f"  ✅ Aggregate executed successfully")
            print(f"     Result: {result}")
        
        # Test distribution
        print("\n  Testing: distribution")
        numeric_cols = context.numeric_cols
        if numeric_cols:
            result = executor.execute_distribution(numeric_cols[0])
            if result and 'error' not in result:
                print(f"  ✅ Distribution executed successfully")
                print(f"     Column: {numeric_cols[0]}")
        
        # Test model_info
        print("\n  Testing: model_info")
        result = executor.execute_model_info()
        if result and 'error' not in result:
            print(f"  ✅ Model info retrieved successfully")
        else:
            print(f"  ⚠️  No model available (expected)")
    
    except Exception as e:
        print(f"❌ Error executing DSL: {e}")

def test_full_pipeline():
    """Test complete chat response pipeline"""
    print("\n" + "="*60)
    print("TEST 5: Full Chat Pipeline")
    print("="*60)
    
    try:
        dataset_id = "018d7cde"
        test_questions = [
            "Tell me about my dataset",
            "What's the average value?",
            "Group by category and show counts",
            "Are there correlations?",
            "Show data distribution",
            "Model performance?",
        ]
        
        for question in test_questions:
            print(f"\n  Question: '{question}'")
            response = get_chat_response(dataset_id, question)
            
            if response.get('status') == 'ok':
                dsl = response.get('dsl', {})
                action = dsl.get('action')
                print(f"  ✅ Response received")
                print(f"     Action: {action}")
                if dsl.get('reason'):
                    print(f"     Reason: {dsl['reason']}")
            else:
                print(f"  ❌ Error: {response}")
    
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║  OmniSearch AI - DSL Chat System Test Suite              ║")
    print("╚" + "="*58 + "╝")
    
    test_intent_detection()
    test_dataset_context()
    test_dsl_builder()
    test_dsl_executor()
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
