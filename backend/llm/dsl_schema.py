"""
STRICT JSON DSL SCHEMA
======================
Defines the ONLY allowed operations for data analysis.
LLM must ONLY return valid JSON conforming to these models.

CORE RULES (NON-NEGOTIABLE):
✅ All queries must validate against these Pydantic models
✅ Max 4 columns per query
✅ Only allowed metrics: mean, sum, count, min, max, median
✅ Only allowed actions: describe, aggregate, groupby, correlation, distribution, model_info
❌ No free text parameters
❌ No inventing columns
❌ No mathematical expressions in LLM
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Literal
from enum import Enum


# ============================================
# ALLOWED METRIC TYPES
# ============================================

class MetricType(str, Enum):
    """Only these metrics are allowed"""
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STD = "std"


class ActionType(str, Enum):
    """Only these DSL actions are allowed"""
    DESCRIBE = "describe"
    AGGREGATE = "aggregate"
    GROUPBY = "groupby"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    MODEL_INFO = "model_info"
    FILTER = "filter"


# ============================================
# BASE DSL MODELS
# ============================================

class BaseDSL(BaseModel):
    """Base class for all DSL queries"""
    action: ActionType = Field(
        ..., 
        description="The operation to perform. MUST be one of: describe, aggregate, groupby, correlation, distribution, model_info"
    )
    
    class Config:
        use_enum_values = False


class DescribeDSL(BaseDSL):
    """Describe dataset schema and structure"""
    action: Literal["describe"] = "describe"
    
    class Config:
        schema_extra = {
            "example": {
                "action": "describe"
            }
        }


class AggregateDSL(BaseDSL):
    """Single aggregate metric over all rows"""
    action: Literal["aggregate"] = "aggregate"
    columns: List[str] = Field(
        ...,
        min_items=1,
        max_items=4,
        description="Numeric columns to aggregate (max 4)"
    )
    metrics: Dict[str, MetricType] = Field(
        ...,
        description="Column → metric mapping. Only allowed: mean, sum, count, min, max, median"
    )
    
    @validator('columns')
    def max_columns(cls, v):
        if len(v) > 4:
            raise ValueError("Maximum 4 columns allowed")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate columns not allowed")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if not v:
            raise ValueError("Must specify at least one metric")
        for col, metric in v.items():
            if not isinstance(metric, str) or metric.lower() not in [m.value for m in MetricType]:
                raise ValueError(f"Invalid metric: {metric}. Allowed: mean, sum, count, min, max, median")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "action": "aggregate",
                "columns": ["price", "quantity"],
                "metrics": {
                    "price": "mean",
                    "quantity": "sum"
                }
            }
        }


class GroupByDSL(BaseDSL):
    """Group by categorical column(s) and aggregate"""
    action: Literal["groupby"] = "groupby"
    group_by: List[str] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Categorical columns to group by (max 3)"
    )
    metrics: Dict[str, MetricType] = Field(
        ...,
        description="Column → metric mapping"
    )
    
    @validator('group_by')
    def max_groupby_cols(cls, v):
        if len(v) > 3:
            raise ValueError("Maximum 3 groupby columns allowed")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate groupby columns not allowed")
        return v
    
    @validator('metrics')
    def validate_metrics(cls, v):
        if not v:
            raise ValueError("Must specify at least one metric")
        for col, metric in v.items():
            if not isinstance(metric, str) or metric.lower() not in [m.value for m in MetricType]:
                raise ValueError(f"Invalid metric: {metric}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "action": "groupby",
                "group_by": ["region", "product"],
                "metrics": {
                    "sales": "sum",
                    "quantity": "mean"
                }
            }
        }


class CorrelationDSL(BaseDSL):
    """Compute correlations between numeric columns"""
    action: Literal["correlation"] = "correlation"
    columns: List[str] = Field(
        ...,
        min_items=2,
        max_items=4,
        description="Numeric columns to correlate (2-4)"
    )
    
    @validator('columns')
    def validate_columns(cls, v):
        if len(v) < 2:
            raise ValueError("Need at least 2 columns for correlation")
        if len(v) > 4:
            raise ValueError("Maximum 4 columns allowed")
        if len(v) != len(set(v)):
            raise ValueError("Duplicate columns not allowed")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "action": "correlation",
                "columns": ["price", "quantity", "revenue"]
            }
        }


class DistributionDSL(BaseDSL):
    """Get distribution stats for a single column"""
    action: Literal["distribution"] = "distribution"
    column: str = Field(
        ...,
        description="Single column to analyze distribution"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "action": "distribution",
                "column": "age"
            }
        }


class ModelInfoDSL(BaseDSL):
    """Get trained model metadata and performance"""
    action: Literal["model_info"] = "model_info"
    
    class Config:
        schema_extra = {
            "example": {
                "action": "model_info"
            }
        }


class FilterCondition(BaseModel):
    """Single filter condition"""
    column: str
    operator: Literal[">", "<", "==", ">=", "<=", "in"]
    value: Optional[List] = None  # For 'in' operator


class FilterDSL(BaseDSL):
    """Filter rows and aggregate"""
    action: Literal["filter"] = "filter"
    conditions: List[FilterCondition] = Field(
        ...,
        min_items=1,
        max_items=3,
        description="Filter conditions (max 3)"
    )
    metrics: Dict[str, MetricType] = Field(
        default={},
        description="Optional aggregations on filtered data"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "action": "filter",
                "conditions": [
                    {"column": "price", "operator": ">", "value": 100},
                    {"column": "region", "operator": "==", "value": "US"}
                ],
                "metrics": {"revenue": "sum"}
            }
        }


# ============================================
# UNION TYPE FOR ALL VALID DSL QUERIES
# ============================================

DSLQuery = (
    DescribeDSL |
    AggregateDSL |
    GroupByDSL |
    CorrelationDSL |
    DistributionDSL |
    ModelInfoDSL |
    FilterDSL
)


class UnsupportedDSL(BaseModel):
    """For queries that cannot be safely executed"""
    action: Literal["unsupported"] = "unsupported"
    reason: str = Field(..., description="Why this query is not supported")
    suggestion: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "action": "unsupported",
                "reason": "Cannot predict future values",
                "suggestion": "Use model_info to see available predictions"
            }
        }


# ============================================
# RESPONSE MODELS
# ============================================

class DSLValidationResponse(BaseModel):
    """Response from DSL validation"""
    valid: bool
    dsl: Optional[Dict] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


class DSLExecutionResponse(BaseModel):
    """Response from DSL execution"""
    status: str  # "ok", "error", "clarification_needed"
    dsl: Dict
    result: Optional[Dict] = None
    confidence: str = "high"  # high, medium, low
    clarification: Optional[Dict] = None  # For ambiguous queries
    explanation: Optional[str] = None
