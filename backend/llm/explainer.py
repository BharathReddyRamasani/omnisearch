"""
RESULT EXPLAINER
================
LLM explains DSL execution results in business language.

CRITICAL RULE:
⚠️ LLM sees RESULT ONLY, never raw data
✅ Safe: {"region": "A": 42.3, "B": 31.7}
❌ Unsafe: Raw row data, raw values

LLM prompt ensures:
- Explain aggregations in business terms
- Highlight significant patterns
- No assumptions, only what data shows
- Deterministic explanations (no hallucination)
"""

import json
from typing import Dict, Any, Optional, List


class ResultExplainer:
    """
    Converts DSL execution results to business explanations using LLM.
    
    LLM operates in "explain only" mode:
    - Sees RESULT only, not raw data
    - No access to data values beyond result
    - No inference about causation
    - No predictions
    """
    
    def __init__(self, use_llm: bool = False, client=None):
        """
        Args:
            use_llm: Use LLM for explanations
            client: OpenAI/Anthropic client
        """
        self.use_llm = use_llm
        self.client = client
    
    def explain(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """
        Generate business explanation for DSL result.
        
        Args:
            dsl: Original DSL query
            result: Execution result
        
        Returns:
            Natural language explanation (or None if no explanation)
        """
        
        if not self.use_llm or not self.client:
            return self._explain_with_heuristics(dsl, result)
        
        return self._explain_with_llm(dsl, result)
    
    def _explain_with_llm(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Use LLM to generate explanation"""
        
        action = dsl.get('action')
        
        # Build explain prompt
        prompt = self._build_explain_prompt(action, dsl, result)
        
        if not prompt:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a business analyst explaining data analysis results.

RULES (NON-NEGOTIABLE):
1. Explain in plain business language
2. Reference ONLY what's in the data result
3. Do NOT make assumptions or inferences beyond data
4. Do NOT claim causation
5. Be concise (2-3 sentences)
6. Do NOT say "based on the data" - just state facts
7. Use specific values from result

Example good explanation:
"Sales in region A average $42.3k, while region B is $31.7k. Region A's sales are 34% higher."

Example bad explanation:
"Region A has better sales because of market conditions..." (assumption)"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return None
    
    def _explain_with_heuristics(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Generate explanation using heuristics (no LLM)"""
        
        action = dsl.get('action')
        
        if action == 'describe':
            return self._explain_describe(result)
        elif action == 'aggregate':
            return self._explain_aggregate(dsl, result)
        elif action == 'groupby':
            return self._explain_groupby(dsl, result)
        elif action == 'distribution':
            return self._explain_distribution(dsl, result)
        elif action == 'correlation':
            return self._explain_correlation(dsl, result)
        
        return None
    
    def _build_explain_prompt(self, action: str, dsl: Dict[str, Any], 
                             result: Dict[str, Any]) -> Optional[str]:
        """Build prompt for LLM explanation"""
        
        result_summary = json.dumps(result, indent=2)[:500]  # First 500 chars
        
        if action == 'aggregate':
            cols = dsl.get('columns', [])
            metrics = dsl.get('metrics', {})
            return f"""Explain this aggregation result in business language.

Query: Calculate {', '.join(metrics.values())} for columns: {', '.join(cols)}

Result:
{result_summary}

Explain what this means in 2-3 sentences."""
        
        elif action == 'groupby':
            group_by = dsl.get('group_by', [])
            metrics = dsl.get('metrics', {})
            return f"""Explain this grouped analysis in business language.

Query: Group by {', '.join(group_by)} and calculate {', '.join(metrics.values())}

Result:
{result_summary}

Explain the key findings in 2-3 sentences."""
        
        elif action == 'distribution':
            col = dsl.get('column', '')
            return f"""Explain this distribution analysis in business language.

Column: {col}

Result:
{result_summary}

Describe the distribution in 2-3 sentences."""
        
        elif action == 'correlation':
            cols = dsl.get('columns', [])
            return f"""Explain these correlations in business language.

Columns: {', '.join(cols)}

Result:
{result_summary}

Highlight the strongest relationships in 2-3 sentences."""
        
        return None
    
    def _explain_describe(self, result: Dict[str, Any]) -> str:
        """Heuristic explanation for describe"""
        
        rows = result.get('rows', 0)
        cols = result.get('columns', [])
        numeric = result.get('numeric_columns', [])
        categorical = result.get('categorical_columns', [])
        
        return (f"This dataset has {rows} rows and {len(cols)} columns: "
                f"{len(numeric)} numeric and {len(categorical)} categorical.")
    
    def _explain_aggregate(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Heuristic explanation for aggregate"""
        
        if 'error' in result:
            return None
        
        metrics = dsl.get('metrics', {})
        lines = []
        
        for col, metric in metrics.items():
            key = f"{col}_{metric}"
            if key in result:
                val = result[key]
                if isinstance(val, float):
                    lines.append(f"{col}: {metric} = {val:.2f}")
                else:
                    lines.append(f"{col}: {metric} = {val}")
        
        return "Aggregation results: " + ", ".join(lines) if lines else None
    
    def _explain_groupby(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Heuristic explanation for groupby"""
        
        if 'error' in result:
            return None
        
        group_by = dsl.get('group_by', [])
        
        # Count distinct groups
        first_key = next(iter(result.keys())) if result else None
        if first_key and isinstance(result[first_key], dict):
            group_count = len(result[first_key])
            return f"Grouped by {', '.join(group_by)}: {group_count} distinct groups found."
        
        return None
    
    def _explain_distribution(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Heuristic explanation for distribution"""
        
        if 'error' in result:
            return None
        
        col = dsl.get('column', '')
        dist_type = result.get('type', '')
        
        if dist_type == 'numeric':
            mean = result.get('mean', 0)
            std = result.get('std', 0)
            min_val = result.get('min', 0)
            max_val = result.get('max', 0)
            
            return (f"Column {col}: mean={mean:.2f}, std={std:.2f}, "
                   f"range=[{min_val:.2f}, {max_val:.2f}]")
        
        elif dist_type == 'categorical':
            unique = result.get('unique_values', 0)
            return f"Column {col}: {unique} unique categorical values."
        
        return None
    
    def _explain_correlation(self, dsl: Dict[str, Any], result: Dict[str, Any]) -> Optional[str]:
        """Heuristic explanation for correlation"""
        
        if 'error' in result:
            return None
        
        corr_matrix = result.get('correlation_matrix', {})
        
        # Find strongest correlations
        strongest = []
        for col1, row in corr_matrix.items():
            for col2, corr in row.items():
                if col1 < col2:  # Avoid duplicates
                    if abs(corr) > 0.7:
                        strongest.append((col1, col2, corr))
        
        if strongest:
            strongest.sort(key=lambda x: abs(x[2]), reverse=True)
            return f"Strongest correlation: {strongest[0][0]} ↔ {strongest[0][1]} ({strongest[0][2]:.3f})"
        
        return "No strong correlations (>0.7) found."


class ConfidenceScorer:
    """
    Assigns confidence scores to DSL execution results.
    
    HIGH: Clear intent, sufficient data, valid metrics
    MEDIUM: Some ambiguity, limited data
    LOW: High uncertainty, edge cases
    """
    
    @staticmethod
    def score(dsl: Dict[str, Any], result: Dict[str, Any], 
             dataset_size: int, data_coverage: float) -> str:
        """
        Score confidence in result.
        
        Args:
            dsl: Original DSL
            result: Execution result
            dataset_size: Number of rows in dataset
            data_coverage: Fraction of data used (0.0-1.0)
        
        Returns:
            "high", "medium", or "low"
        """
        
        # If error, confidence is low
        if 'error' in result:
            return 'low'
        
        # Check result size
        if isinstance(result, dict):
            if not result:
                return 'low'
        
        # Check data coverage
        if data_coverage < 0.1:  # Less than 10% of data
            return 'low'
        
        if data_coverage < 0.5:  # Less than 50%
            return 'medium'
        
        # Check action
        action = dsl.get('action')
        
        # Distribution on small datasets has lower confidence
        if action == 'distribution' and dataset_size < 100:
            return 'medium'
        
        # Correlation with few datapoints has lower confidence
        if action == 'correlation' and dataset_size < 30:
            return 'medium'
        
        # Otherwise high confidence
        return 'high'
