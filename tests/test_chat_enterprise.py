"""
COMPREHENSIVE CHAT TESTING
===========================
Test all phases of the enterprise chat architecture.

Run with: pytest tests/test_chat_enterprise.py -v
"""

import pytest
import json
from typing import Dict, Any
from backend.chat.orchestrator import ChatOrchestrator
from backend.llm.validator import DSLValidator
from backend.llm.planner import LLMPlanner
from backend.chat.clarifier import IntentClarifier
from backend.rag.index import RAGIndexBuilder


# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def sample_dataset_id():
    """Return a valid dataset ID for testing"""
    # Assuming you have at least one dataset
    return "00f53e8a"  # From workspace


@pytest.fixture
def orchestrator(sample_dataset_id):
    """Initialize ChatOrchestrator"""
    return ChatOrchestrator(
        dataset_id=sample_dataset_id,
        use_llm=False,  # Use heuristics for deterministic testing
        use_rag=True,
        use_explanation=True
    )


# ============================================
# PHASE 1: DSL SCHEMA & VALIDATION TESTS
# ============================================

class TestDSLSchema:
    """Test strict DSL schema validation"""
    
    def test_valid_aggregate_dsl(self):
        """Valid aggregate DSL should pass"""
        from backend.llm.dsl_schema import AggregateDSL
        
        dsl = AggregateDSL(
            columns=["price", "quantity"],
            metrics={"price": "mean", "quantity": "sum"}
        )
        
        assert dsl.action == "aggregate"
        assert len(dsl.columns) == 2
    
    def test_max_columns_constraint(self):
        """Exceed max 4 columns should fail"""
        from backend.llm.dsl_schema import AggregateDSL
        
        with pytest.raises(ValueError):
            AggregateDSL(
                columns=["a", "b", "c", "d", "e"],  # 5 columns
                metrics={"a": "mean"}
            )
    
    def test_invalid_metric_type(self):
        """Invalid metric should fail"""
        from backend.llm.dsl_schema import AggregateDSL
        
        with pytest.raises(ValueError):
            AggregateDSL(
                columns=["price"],
                metrics={"price": "invalid_metric"}
            )
    
    def test_groupby_max_columns(self):
        """Exceed max 3 groupby columns should fail"""
        from backend.llm.dsl_schema import GroupByDSL
        
        with pytest.raises(ValueError):
            GroupByDSL(
                group_by=["a", "b", "c", "d"],  # 4 columns
                metrics={"value": "sum"}
            )


class TestDSLValidator:
    """Test schema-aware DSL validation"""
    
    def test_column_existence(self, orchestrator):
        """Non-existent column should be rejected"""
        dsl = {
            'action': 'aggregate',
            'columns': ['nonexistent_column'],
            'metrics': {'nonexistent_column': 'mean'}
        }
        
        response = orchestrator.validator.validate(dsl)
        assert not response.valid
        assert any('not found' in err.lower() for err in response.errors)
    
    def test_numeric_metric_on_categorical(self, orchestrator):
        """Mean on categorical column should be rejected"""
        # Get first categorical column
        cat_col = orchestrator.context.categorical_cols[0]
        
        dsl = {
            'action': 'aggregate',
            'columns': [cat_col],
            'metrics': {cat_col: 'mean'}
        }
        
        response = orchestrator.validator.validate(dsl)
        assert not response.valid
    
    def test_valid_aggregate(self, orchestrator):
        """Valid aggregate should pass"""
        if not orchestrator.context.numeric_cols:
            pytest.skip("No numeric columns")
        
        col = orchestrator.context.numeric_cols[0]
        dsl = {
            'action': 'aggregate',
            'columns': [col],
            'metrics': {col: 'mean'}
        }
        
        response = orchestrator.validator.validate(dsl)
        assert response.valid
    
    def test_groupby_needs_categorical(self, orchestrator):
        """Groupby should prefer categorical columns"""
        if not orchestrator.context.categorical_cols:
            pytest.skip("No categorical columns")
        
        col = orchestrator.context.categorical_cols[0]
        dsl = {
            'action': 'groupby',
            'group_by': [col],
            'metrics': {'count_col': 'count'}
        }
        
        response = orchestrator.validator.validate(dsl)
        assert response.valid


# ============================================
# PHASE 2: LLM PLANNER TESTS
# ============================================

class TestLLMPlanner:
    """Test heuristic DSL planning"""
    
    def test_describe_intent(self, orchestrator):
        """Describe query should generate describe DSL"""
        planner = orchestrator.planner
        
        dsl = planner.plan("Tell me about this dataset")
        
        assert dsl.get('action') == 'describe'
    
    def test_aggregate_intent(self, orchestrator):
        """Aggregate query should generate aggregate DSL"""
        if not orchestrator.context.numeric_cols:
            pytest.skip("No numeric columns")
        
        planner = orchestrator.planner
        col = orchestrator.context.numeric_cols[0]
        
        dsl = planner.plan(f"What is the average {col}?")
        
        assert dsl.get('action') == 'aggregate'
        assert len(dsl.get('columns', [])) > 0
    
    def test_groupby_intent(self, orchestrator):
        """Groupby query should generate groupby DSL"""
        if not orchestrator.context.categorical_cols or not orchestrator.context.numeric_cols:
            pytest.skip("Need both categorical and numeric columns")
        
        planner = orchestrator.planner
        
        dsl = planner.plan("How do sales vary by region?")
        
        assert dsl.get('action') == 'groupby'
    
    def test_distribution_intent(self, orchestrator):
        """Distribution query should generate distribution DSL"""
        planner = orchestrator.planner
        
        dsl = planner.plan("What is the distribution of values?")
        
        assert dsl.get('action') == 'distribution'


# ============================================
# PHASE 3: INTENT CLARIFICATION TESTS
# ============================================

class TestIntentClarifier:
    """Test ambiguity detection and clarification"""
    
    def test_clarifier_initialization(self, orchestrator):
        """Clarifier should initialize without errors"""
        assert orchestrator.clarifier is not None
    
    def test_clarify_unambiguous_query(self, orchestrator):
        """Clear query should not need clarification"""
        dsl = {'action': 'describe'}
        
        clarification = orchestrator.clarifier.clarify("describe dataset", dsl)
        
        assert clarification is None


# ============================================
# PHASE 4: ORCHESTRATOR TESTS
# ============================================

class TestChatOrchestrator:
    """Test end-to-end chat orchestration"""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Orchestrator should initialize"""
        assert orchestrator.context is not None
        assert orchestrator.planner is not None
        assert orchestrator.validator is not None
        assert orchestrator.executor is not None
    
    def test_simple_describe_chat(self, orchestrator):
        """Simple describe chat should succeed"""
        response = orchestrator.chat("Tell me about the dataset")
        
        assert response['status'] == 'ok'
        assert response['dsl']['action'] == 'describe'
        assert 'result' in response
    
    def test_aggregate_chat(self, orchestrator):
        """Aggregate chat should succeed"""
        if not orchestrator.context.numeric_cols:
            pytest.skip("No numeric columns")
        
        col = orchestrator.context.numeric_cols[0]
        response = orchestrator.chat(f"Average {col}?")
        
        assert response['status'] == 'ok'
        assert response['dsl']['action'] == 'aggregate'
    
    def test_error_handling(self, orchestrator):
        """Invalid query should return error"""
        dsl = {
            'action': 'aggregate',
            'columns': ['nonexistent'],
            'metrics': {'nonexistent': 'mean'}
        }
        
        # Direct execution would fail, but orchestrator handles gracefully
        response = orchestrator._execute_dsl(dsl)
        
        assert 'error' in response
    
    def test_confidence_scoring(self, orchestrator):
        """Response should include confidence score"""
        response = orchestrator.chat("Describe this dataset")
        
        assert 'confidence' in response
        assert response['confidence'] in ['high', 'medium', 'low']
    
    def test_audit_logging(self, orchestrator):
        """Chat should log to audit log"""
        response = orchestrator.chat("Describe dataset")
        
        assert 'audit_id' in response


# ============================================
# PHASE 5: RAG TESTS
# ============================================

class TestRAGIndex:
    """Test RAG knowledge base building"""
    
    def test_rag_builder_initialization(self, sample_dataset_id):
        """RAG builder should initialize"""
        builder = RAGIndexBuilder(sample_dataset_id)
        assert builder.df is not None
    
    def test_rag_index_building(self, sample_dataset_id):
        """RAG should build chunks"""
        builder = RAGIndexBuilder(sample_dataset_id)
        chunks = builder.build_index()
        
        assert len(chunks) > 0
        assert any(c.chunk_type == 'column_def' for c in chunks)
        assert any(c.chunk_type == 'eda_summary' for c in chunks)


# ============================================
# GOLDEN TEST CASES
# ============================================

class TestGoldenQueries:
    """Test golden query set for determinism"""
    
    @pytest.fixture
    def golden_tests(self):
        """Load golden tests from JSON"""
        with open('tests/chat_queries.json', 'r') as f:
            data = json.load(f)
        return data['golden_tests']
    
    def test_determinism(self, orchestrator, golden_tests):
        """Same query should always produce same DSL"""
        query = "Tell me about this dataset"
        
        dsl1 = orchestrator.planner.plan(query)
        dsl2 = orchestrator.planner.plan(query)
        dsl3 = orchestrator.planner.plan(query)
        
        # Check that actions are identical
        assert dsl1.get('action') == dsl2.get('action')
        assert dsl2.get('action') == dsl3.get('action')
    
    def test_golden_describe(self, orchestrator):
        """Test golden describe query"""
        response = orchestrator.chat("Tell me about this dataset")
        
        assert response['status'] == 'ok'
        assert response['dsl']['action'] == 'describe'
    
    def test_golden_aggregate(self, orchestrator):
        """Test golden aggregate query"""
        if not orchestrator.context.numeric_cols:
            pytest.skip("No numeric columns")
        
        col = orchestrator.context.numeric_cols[0]
        response = orchestrator.chat(f"What is the average {col}?")
        
        assert response['status'] == 'ok'
        assert response['dsl']['action'] == 'aggregate'


# ============================================
# SAFETY & SECURITY TESTS
# ============================================

class TestSafety:
    """Test safety constraints"""
    
    def test_no_future_prediction(self, orchestrator):
        """Predict future should be unsupported"""
        dsl = orchestrator.planner.plan("Predict next month sales")
        
        assert dsl.get('action') == 'unsupported'
    
    def test_column_limit(self, orchestrator):
        """Query should not exceed column limits"""
        if len(orchestrator.context.numeric_cols) < 4:
            pytest.skip("Not enough columns for test")
        
        # Planner should limit to 4 columns
        dsl = orchestrator.planner.plan("correlate all numeric columns")
        
        if dsl.get('action') == 'correlation':
            assert len(dsl.get('columns', [])) <= 4


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
