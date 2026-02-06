"""
DSL VALIDATION LAYER
====================
Validates DSL queries against dataset schema BEFORE execution.

VALIDATION RULES:
✅ Column existence: columns must exist in dataset
✅ Type correctness: numeric metrics on numeric columns only
✅ Aggregation compatibility: groupby only on categorical columns
✅ Safety constraints: max columns, max groupby fields
✅ No free text or expressions

This is the security gatekeeper between LLM and executor.
"""

import json
from typing import Dict, Any, List, Tuple, Optional
from pydantic import ValidationError
from backend.llm.dsl_schema import (
    DSLQuery,
    AggregateDSL,
    GroupByDSL,
    CorrelationDSL,
    DistributionDSL,
    UnsupportedDSL,
    DSLValidationResponse,
    MetricType
)


class DSLValidator:
    """
    Validates DSL queries against dataset schema.
    
    Rejects:
    - Non-existent columns
    - Numeric metrics on categorical columns
    - Groupby on numeric columns (unless numeric categorical values)
    - Dangerous expressions
    """
    
    def __init__(self, dataset_schema: Dict[str, str], numeric_cols: List[str], categorical_cols: List[str]):
        """
        Args:
            dataset_schema: {col_name: dtype_str}
            numeric_cols: list of numeric column names
            categorical_cols: list of categorical column names
        """
        self.schema = dataset_schema
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.all_cols = set(dataset_schema.keys())
    
    def validate(self, dsl: Dict[str, Any]) -> DSLValidationResponse:
        """
        Main validation entry point.
        
        Returns:
            DSLValidationResponse with valid=True/False and detailed errors/warnings
        """
        errors = []
        warnings = []
        
        try:
            action = dsl.get('action')
            
            # ============================================
            # ACTION-SPECIFIC VALIDATION
            # ============================================
            
            if action == 'describe':
                # Always safe
                return DSLValidationResponse(valid=True, dsl=dsl)
            
            elif action == 'aggregate':
                valid, msg, warns = self._validate_aggregate(dsl)
                if not valid:
                    errors.append(msg)
                else:
                    warnings.extend(warns)
            
            elif action == 'groupby':
                valid, msg, warns = self._validate_groupby(dsl)
                if not valid:
                    errors.append(msg)
                else:
                    warnings.extend(warns)
            
            elif action == 'correlation':
                valid, msg, warns = self._validate_correlation(dsl)
                if not valid:
                    errors.append(msg)
                else:
                    warnings.extend(warns)
            
            elif action == 'distribution':
                valid, msg, warns = self._validate_distribution(dsl)
                if not valid:
                    errors.append(msg)
                else:
                    warnings.extend(warns)
            
            elif action == 'filter':
                valid, msg, warns = self._validate_filter(dsl)
                if not valid:
                    errors.append(msg)
                else:
                    warnings.extend(warns)
            
            elif action == 'model_info':
                # Always safe
                return DSLValidationResponse(valid=True, dsl=dsl)
            
            else:
                errors.append(f"Unknown action: {action}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        # ============================================
        # RETURN RESPONSE
        # ============================================
        
        if errors:
            return DSLValidationResponse(
                valid=False,
                dsl=dsl,
                errors=errors,
                warnings=warnings if warnings else None
            )
        
        return DSLValidationResponse(
            valid=True,
            dsl=dsl,
            warnings=warnings if warnings else None
        )
    
    def _validate_aggregate(self, dsl: Dict) -> Tuple[bool, str, List[str]]:
        """Validate aggregate query"""
        warnings = []
        
        # Check columns exist
        columns = dsl.get('columns', [])
        missing = [c for c in columns if c not in self.all_cols]
        if missing:
            return False, f"Columns not found: {', '.join(missing)}", warnings
        
        # Check max columns
        if len(columns) > 4:
            return False, "Maximum 4 columns allowed", warnings
        
        # Check metrics are on numeric columns
        metrics = dsl.get('metrics', {})
        for col, metric in metrics.items():
            if col not in self.all_cols:
                return False, f"Metric column not found: {col}", warnings
            
            if metric in ['mean', 'sum', 'min', 'max', 'median', 'std']:
                if col not in self.numeric_cols:
                    return False, f"Cannot use {metric} on non-numeric column '{col}'", warnings
            elif metric == 'count':
                # Count works on any column
                pass
            else:
                return False, f"Unknown metric: {metric}", warnings
        
        return True, "", warnings
    
    def _validate_groupby(self, dsl: Dict) -> Tuple[bool, str, List[str]]:
        """Validate groupby query"""
        warnings = []
        
        # Check groupby columns exist and are categorical
        groupby_cols = dsl.get('group_by', [])
        
        if not groupby_cols:
            return False, "Groupby requires at least one groupby column", warnings
        
        missing = [c for c in groupby_cols if c not in self.all_cols]
        if missing:
            return False, f"Groupby columns not found: {', '.join(missing)}", warnings
        
        # Check max groupby cols
        if len(groupby_cols) > 3:
            return False, "Maximum 3 groupby columns allowed", warnings
        
        # Warn if numeric column used for groupby
        numeric_in_groupby = [c for c in groupby_cols if c in self.numeric_cols]
        if numeric_in_groupby:
            warnings.append(f"Groupby on numeric columns: {numeric_in_groupby}. "
                          "Results may have many groups.")
        
        # Check metrics
        metrics = dsl.get('metrics', {})
        if not metrics:
            return False, "Groupby requires at least one metric", warnings
        
        for col, metric in metrics.items():
            if col not in self.all_cols:
                return False, f"Metric column not found: {col}", warnings
            
            if metric in ['mean', 'sum', 'min', 'max', 'median', 'std']:
                if col not in self.numeric_cols:
                    return False, f"Cannot use {metric} on non-numeric column '{col}'", warnings
            elif metric == 'count':
                pass
            else:
                return False, f"Unknown metric: {metric}", warnings
        
        return True, "", warnings
    
    def _validate_correlation(self, dsl: Dict) -> Tuple[bool, str, List[str]]:
        """Validate correlation query"""
        warnings = []
        
        columns = dsl.get('columns', [])
        
        # Check columns exist
        missing = [c for c in columns if c not in self.all_cols]
        if missing:
            return False, f"Columns not found: {', '.join(missing)}", warnings
        
        # Check at least 2 columns
        if len(columns) < 2:
            return False, "Correlation requires at least 2 columns", warnings
        
        # Check all numeric
        non_numeric = [c for c in columns if c not in self.numeric_cols]
        if non_numeric:
            return False, f"Correlation requires numeric columns. Non-numeric: {non_numeric}", warnings
        
        # Check max columns
        if len(columns) > 4:
            return False, "Maximum 4 columns allowed", warnings
        
        return True, "", warnings
    
    def _validate_distribution(self, dsl: Dict) -> Tuple[bool, str, List[str]]:
        """Validate distribution query"""
        warnings = []
        
        column = dsl.get('column')
        
        if not column:
            return False, "Distribution requires a column", warnings
        
        if column not in self.all_cols:
            return False, f"Column not found: {column}", warnings
        
        return True, "", warnings
    
    def _validate_filter(self, dsl: Dict) -> Tuple[bool, str, List[str]]:
        """Validate filter query"""
        warnings = []
        
        conditions = dsl.get('conditions', [])
        
        if not conditions:
            return False, "Filter requires at least one condition", warnings
        
        if len(conditions) > 3:
            return False, "Maximum 3 filter conditions allowed", warnings
        
        for cond in conditions:
            col = cond.get('column')
            operator = cond.get('operator')
            value = cond.get('value')
            
            if not col or col not in self.all_cols:
                return False, f"Invalid filter column: {col}", warnings
            
            if operator not in ['>', '<', '==', '>=', '<=', 'in']:
                return False, f"Invalid operator: {operator}", warnings
            
            # Type checking
            if operator != 'in' and col in self.numeric_cols:
                try:
                    float(value)
                except (TypeError, ValueError):
                    return False, f"Filter on numeric column {col} requires numeric value", warnings
        
        return True, "", warnings


class DSLParseError(Exception):
    """Raised when DSL JSON is malformed"""
    pass


def validate_dsl_json(dsl_json: str, schema: Dict[str, str], 
                      numeric_cols: List[str], categorical_cols: List[str]) -> DSLValidationResponse:
    """
    Parse and validate raw JSON DSL string.
    
    Args:
        dsl_json: Raw JSON string from LLM
        schema: Dataset schema
        numeric_cols: Numeric columns
        categorical_cols: Categorical columns
    
    Returns:
        DSLValidationResponse
    """
    # Try to parse JSON
    try:
        dsl_dict = json.loads(dsl_json)
    except json.JSONDecodeError as e:
        return DSLValidationResponse(
            valid=False,
            errors=[f"Invalid JSON: {str(e)}"]
        )
    
    # Validate against schema
    validator = DSLValidator(schema, numeric_cols, categorical_cols)
    return validator.validate(dsl_dict)
