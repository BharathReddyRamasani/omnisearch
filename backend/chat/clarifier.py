"""
INTENT CLARIFICATION LAYER
===========================
Detects ambiguous queries and asks for clarification.

WHEN TO CLARIFY:
✅ Missing column specification ("sales" - which one?)
✅ Ambiguous metrics ("show me sales" - mean, sum, or count?)
✅ Unclear grouping ("by region" - which region column?)
✅ Multiple valid interpretations
❌ Never clarify obvious queries
❌ Never block valid queries

Returns:
{
  "status": "clarification_needed",
  "options": [
    {"interpretation": "...", "dsl": {...}},
    ...
  ]
}
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from backend.services.utils import load_clean, load_raw


class IntentClarifier:
    """
    Detects and resolves query ambiguity intelligently.
    """
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self._load_schema()
    
    def _load_schema(self):
        """Load dataset schema"""
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
    
    def clarify(self, query: str, proposed_dsl: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if query is ambiguous and provide clarification options.
        
        Returns:
            None if clear (proceed with proposed DSL)
            {"status": "clarification_needed", "options": [...]} if ambiguous
        """
        
        # Check for ambiguities
        action = proposed_dsl.get('action')
        
        if action == 'aggregate':
            return self._clarify_aggregate(query, proposed_dsl)
        
        elif action == 'groupby':
            return self._clarify_groupby(query, proposed_dsl)
        
        elif action == 'distribution':
            return self._clarify_distribution(query, proposed_dsl)
        
        # No clarification needed
        return None
    
    def _clarify_aggregate(self, query: str, dsl: Dict) -> Optional[Dict]:
        """Check if aggregate query is ambiguous"""
        
        columns = dsl.get('columns', [])
        metrics = dsl.get('metrics', {})
        
        # AMBIGUITY 1: Multiple valid metrics
        # "show me sales" → could be mean, sum, or count
        if len(columns) == 1 and len(metrics) == 1:
            col = columns[0]
            current_metric = list(metrics.values())[0]
            
            # If metric was defaulted, offer alternatives
            if current_metric == 'mean' and col in self.numeric_cols:
                options = self._generate_metric_options(col, query)
                if len(options) > 1:
                    return {
                        'status': 'clarification_needed',
                        'type': 'metric_ambiguity',
                        'question': f'For "{col}", do you want {self._format_metric_list(options)}?',
                        'options': options
                    }
        
        return None
    
    def _clarify_groupby(self, query: str, dsl: Dict) -> Optional[Dict]:
        """Check if groupby query is ambiguous"""
        
        groupby_cols = dsl.get('group_by', [])
        metrics = dsl.get('metrics', {})
        
        # AMBIGUITY 1: Multiple categorical columns match
        # "by region" could match multiple columns
        possible_cols = self._find_similar_columns(query, self.categorical_cols)
        if len(possible_cols) > 1 and len(groupby_cols) == 1:
            return {
                'status': 'clarification_needed',
                'type': 'column_ambiguity',
                'question': f'Which grouping column did you mean: {", ".join(possible_cols)}?',
                'options': [
                    {
                        'interpretation': f'Group by {col}',
                        'dsl': {
                            'action': 'groupby',
                            'group_by': [col],
                            'metrics': metrics
                        }
                    }
                    for col in possible_cols
                ]
            }
        
        # AMBIGUITY 2: Multiple metric interpretations
        metric_col = list(metrics.keys())[0] if metrics else None
        if metric_col:
            options = self._generate_metric_options(metric_col, query)
            if len(options) > 1:
                return {
                    'status': 'clarification_needed',
                    'type': 'metric_ambiguity',
                    'question': f'For aggregation, do you want: {self._format_metric_list(options)}?',
                    'options': [
                        {
                            'interpretation': f'Sum of {metric_col}',
                            'dsl': {
                                'action': 'groupby',
                                'group_by': groupby_cols,
                                'metrics': {metric_col: opt}
                            }
                        }
                        for opt in options
                    ]
                }
        
        return None
    
    def _clarify_distribution(self, query: str, dsl: Dict) -> Optional[Dict]:
        """Check if distribution query is ambiguous"""
        
        possible_cols = self._find_similar_columns(query, self.columns)
        
        if len(possible_cols) > 1:
            return {
                'status': 'clarification_needed',
                'type': 'column_ambiguity',
                'question': f'Which column did you mean: {", ".join(possible_cols[:5])}?',
                'options': [
                    {
                        'interpretation': f'Distribution of {col}',
                        'dsl': {
                            'action': 'distribution',
                            'column': col
                        }
                    }
                    for col in possible_cols[:5]
                ]
            }
        
        return None
    
    def _generate_metric_options(self, column: str, query: str) -> List[str]:
        """Generate plausible metric options based on context"""
        
        options = []
        query_lower = query.lower()
        
        # Check what's mentioned explicitly
        if 'mean' in query_lower or 'average' in query_lower or 'avg' in query_lower:
            options.append('mean')
        
        if 'sum' in query_lower or 'total' in query_lower:
            options.append('sum')
        
        if 'count' in query_lower or 'how many' in query_lower:
            options.append('count')
        
        if 'min' in query_lower or 'minimum' in query_lower:
            options.append('min')
        
        if 'max' in query_lower or 'maximum' in query_lower:
            options.append('max')
        
        if 'median' in query_lower or 'middle' in query_lower:
            options.append('median')
        
        # If nothing explicit, offer common options
        if not options:
            if column in self.numeric_cols:
                options = ['mean', 'sum', 'count']
            else:
                options = ['count']
        
        return list(dict.fromkeys(options))  # Remove duplicates, preserve order
    
    def _find_similar_columns(self, query: str, candidates: List[str]) -> List[str]:
        """Find columns that match query keywords"""
        
        # Extract potential column names from query
        words = re.findall(r'\b\w+\b', query.lower())
        
        matches = []
        for col in candidates:
            col_lower = col.lower()
            col_words = re.findall(r'\b\w+\b', col_lower)
            
            # Check for exact or partial matches
            if any(word in col_lower for word in words):
                matches.append(col)
        
        return matches if matches else candidates[:3]
    
    def _format_metric_list(self, metrics: List[str]) -> str:
        """Format metric list for human reading"""
        if not metrics:
            return ""
        if len(metrics) == 1:
            return metrics[0]
        if len(metrics) == 2:
            return f"{metrics[0]} or {metrics[1]}"
        return ", ".join(metrics[:-1]) + f", or {metrics[-1]}"


class ClarificationHandler:
    """
    Converts user's clarification response back to DSL.
    """
    
    @staticmethod
    def resolve_clarification(clarification_response: str, 
                            clarification_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user's response to clarification.
        
        Args:
            clarification_response: User's choice (e.g., "mean" or "0" for first option)
            clarification_context: Original clarification dict with options
        
        Returns:
            Resolved DSL
        """
        
        options = clarification_context.get('options', [])
        
        if not options:
            return {'action': 'unsupported', 'reason': 'No clarification options available'}
        
        # Try to match user response to an option
        response_lower = clarification_response.lower().strip()
        
        # Try numeric index (0, 1, 2, ...)
        if response_lower.isdigit():
            idx = int(response_lower)
            if 0 <= idx < len(options):
                return options[idx].get('dsl', {})
        
        # Try matching option interpretation
        for opt in options:
            interp = opt.get('interpretation', '').lower()
            if response_lower in interp or interp in response_lower:
                return opt.get('dsl', {})
        
        return {'action': 'unsupported', 'reason': 'Could not match clarification response'}
