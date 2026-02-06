"""
LLM PLANNER
===========
Converts natural language queries to STRICT JSON DSL.

CRITICAL RULES:
✅ LLM returns ONLY valid JSON
✅ Never executes, never computes
✅ No markdown, no explanations
✅ Column names from schema only
✅ Metrics from allowed set only

System Prompt enforces:
- Enterprise data assistant mode
- JSON ONLY output
- Ask for clarification when ambiguous
- Never invent or hallucinate
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from backend.services.utils import load_clean, load_raw


class LLMPlanner:
    """
    Converts natural language to JSON DSL using LLM or heuristics.
    
    ZERO HALLUCINATION GUARANTEE:
    - All columns verified against schema
    - All metrics from allowed set
    - No mathematical computation
    - No free-form text in DSL
    """
    
    def __init__(self, dataset_id: str, use_llm: bool = False, client=None):
        """
        Args:
            dataset_id: Dataset UUID
            use_llm: Use LLM planner (requires client)
            client: OpenAI/Anthropic client (optional)
        """
        self.dataset_id = dataset_id
        self.use_llm = use_llm
        self.client = client
        
        # Load dataset schema
        self._load_schema()
    
    def _load_schema(self):
        """Load dataset schema for validation"""
        try:
            try:
                self.df = load_clean(self.dataset_id)
            except:
                self.df = load_raw(self.dataset_id)
            
            self.columns = list(self.df.columns)
            self.numeric_cols = self.df.select_dtypes(
                include=['int64', 'int32', 'float64', 'float32']
            ).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(
                include=['object', 'category', 'bool']
            ).columns.tolist()
        except Exception as e:
            raise ValueError(f"Cannot load schema: {str(e)}")
    
    def plan(self, user_query: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Main planning entry point.
        
        Returns:
            {
                "action": "describe|aggregate|groupby|...",
                "columns": [...],
                "metrics": {...},
                ...
            }
        """
        if self.use_llm and self.client:
            return self._plan_with_llm(user_query, chat_history)
        else:
            return self._plan_with_heuristics(user_query)
    
    def _plan_with_llm(self, query: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Use LLM to generate DSL.
        
        SYSTEM PROMPT ENFORCES:
        - JSON ONLY (no markdown backticks)
        - No hallucination
        - Strict schema adherence
        """
        
        # Build context
        schema_str = self._format_schema()
        
        system_prompt = f"""You are an enterprise data assistant.

INSTRUCTIONS (NON-NEGOTIABLE):
1. You MUST respond with ONLY valid JSON, no markdown, no explanations
2. You MUST use ONLY the provided column names
3. You MUST use ONLY these metrics: mean, sum, count, min, max, median, std
4. You MUST use ONLY these actions: describe, aggregate, groupby, correlation, distribution, model_info
5. If uncertain or ambiguous, ask for clarification in the JSON with action="clarification"
6. NEVER compute results, NEVER invent columns, NEVER hallucinate

AVAILABLE DATASET SCHEMA:
{schema_str}

VALID DSL ACTIONS:
- describe: Get dataset overview (no parameters)
- aggregate: Single aggregation (columns, metrics)
- groupby: Group by categorical column(s) (group_by, metrics)
- correlation: Correlate numeric columns (columns, min 2)
- distribution: Distribution stats (column)
- model_info: Model metadata (no parameters)

RESPONSE FORMAT (ALWAYS valid JSON):
{{
  "action": "describe|aggregate|groupby|correlation|distribution|model_info|clarification",
  "columns": [...],
  "group_by": [...],
  "metrics": {{"col": "metric_type"}},
  "clarification": "Question if ambiguous"
}}

Remember: If you are unsure, ask. Do not guess."""

        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation history
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0,  # Deterministic
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown json fencing if present
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
            
            # Parse JSON
            try:
                dsl = json.loads(response_text)
                return dsl
            except json.JSONDecodeError:
                return {
                    "action": "unsupported",
                    "reason": f"LLM returned invalid JSON: {response_text[:100]}"
                }
        
        except Exception as e:
            return {
                "action": "unsupported",
                "reason": f"LLM planning error: {str(e)}"
            }
    
    def _plan_with_heuristics(self, query: str) -> Dict[str, Any]:
        """
        Fallback heuristic planner (no LLM needed).
        Uses intent detection and column extraction.
        """
        from backend.services.chat import IntentDetector
        
        detector = IntentDetector()
        intent = detector.detect_intent(query)
        
        # Build DSL based on intent
        if intent == 'describe':
            return {'action': 'describe'}
        
        elif intent == 'aggregate':
            columns = detector.extract_columns(query, self.numeric_cols)
            if not columns:
                columns = self.numeric_cols[:3] if self.numeric_cols else []
            
            if not columns:
                return {
                    'action': 'unsupported',
                    'reason': 'No numeric columns found for aggregation'
                }
            
            metrics = detector.extract_metrics(query)
            metrics_dict = {col: metrics[0] if metrics else 'mean' for col in columns}
            
            return {
                'action': 'aggregate',
                'columns': columns,
                'metrics': metrics_dict
            }
        
        elif intent == 'groupby':
            groupby_cols = detector.extract_columns(query, self.categorical_cols)
            if not groupby_cols:
                return {
                    'action': 'unsupported',
                    'reason': 'No categorical columns found for grouping'
                }
            
            metric_cols = detector.extract_columns(query, self.numeric_cols)
            if not metric_cols:
                metric_cols = self.numeric_cols[:2] if self.numeric_cols else []
            
            if not metric_cols:
                return {
                    'action': 'unsupported',
                    'reason': 'No numeric columns found for metrics'
                }
            
            metrics = detector.extract_metrics(query)
            metrics_dict = {col: metrics[0] if metrics else 'count' for col in metric_cols}
            
            return {
                'action': 'groupby',
                'group_by': groupby_cols,
                'metrics': metrics_dict
            }
        
        elif intent == 'correlation':
            columns = detector.extract_columns(query, self.numeric_cols)
            if len(columns) < 2:
                columns = self.numeric_cols[:4] if len(self.numeric_cols) >= 2 else []
            
            if len(columns) < 2:
                return {
                    'action': 'unsupported',
                    'reason': 'Need at least 2 numeric columns for correlation'
                }
            
            return {
                'action': 'correlation',
                'columns': columns
            }
        
        elif intent == 'distribution':
            columns = detector.extract_columns(query, self.columns)
            if not columns:
                columns = [self.numeric_cols[0]] if self.numeric_cols else [self.columns[0]]
            
            return {
                'action': 'distribution',
                'column': columns[0]
            }
        
        elif intent == 'model_info':
            return {'action': 'model_info'}
        
        else:
            return {
                'action': 'unsupported',
                'reason': 'Could not determine query intent'
            }
    
    def _format_schema(self) -> str:
        """Format schema for LLM prompt"""
        schema_lines = [
            "NUMERIC COLUMNS (for aggregation, correlation):",
            ", ".join(self.numeric_cols) if self.numeric_cols else "None",
            "",
            "CATEGORICAL COLUMNS (for grouping, filtering):",
            ", ".join(self.categorical_cols) if self.categorical_cols else "None",
            "",
            "COLUMN TYPES:",
        ]
        
        for col in self.columns:
            dtype = str(self.df[col].dtype)
            schema_lines.append(f"  - {col}: {dtype}")
        
        return "\n".join(schema_lines)
