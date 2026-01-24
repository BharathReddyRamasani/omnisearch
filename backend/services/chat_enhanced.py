"""
SCHEMA-AWARE CHAT SYSTEM (NOT RAG):
✅ Strict JSON DSL parsing + validation
✅ Column impact explanations with evidence
✅ Feature importance integration
❌ No FAISS indexing (would add RAG)
❌ No embeddings (not needed for DSL)
❌ No retrieval system (schema-aware, not semantic)

Design: Single DSL-based approach (strict JSON)
Users submit structured JSON queries, not natural language
"""
import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from backend.services.utils import model_dir, load_raw, load_clean


class ColumnImpactAnalyzer:
    """
    Explain how features impact predictions with computed evidence.
    """
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.feature_importance = {}
        self.load_feature_importance()
    
    def load_feature_importance(self):
        """Load feature importance from trained model"""
        try:
            meta_path = os.path.join(model_dir(self.dataset_id), "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    fi = meta.get("feature_importance", {})
                    if isinstance(fi, dict):
                        self.feature_importance = fi.get("scores", {})
        except:
            self.feature_importance = {}
    
    def explain_column_impact(self, column: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain how a specific column impacts the model.
        Returns impact score, trend, and evidence.
        """
        try:
            col_data = df[column].dropna()
            if len(col_data) == 0:
                return {"error": f"No valid data for column {column}"}
            
            # Get importance score
            importance = self.feature_importance.get(column, 0.0)
            
            # Determine impact level
            if importance > 0.15:
                impact_level = "HIGH"
            elif importance > 0.05:
                impact_level = "MEDIUM"
            else:
                impact_level = "LOW"
            
            # Compute evidence
            evidence = {
                "importance_score": round(float(importance), 6),
                "impact_level": impact_level,
                "data_points": int(len(col_data)),
                "missing_count": int(col_data.isna().sum())
            }
            
            # Add statistical evidence
            if pd.api.types.is_numeric_dtype(col_data):
                evidence.update({
                    "type": "numeric",
                    "mean": round(float(col_data.mean()), 4),
                    "std": round(float(col_data.std()), 4),
                    "range": [round(float(col_data.min()), 4), round(float(col_data.max()), 4)],
                    "correlation_with_target": "computed from model training"
                })
            else:
                evidence.update({
                    "type": "categorical",
                    "unique_values": int(col_data.nunique()),
                    "top_values": col_data.value_counts().head(3).to_dict()
                })
            
            return {
                "column": column,
                "impact_level": impact_level,
                "importance_score": round(float(importance), 6),
                "evidence": evidence
            }
        
        except Exception as e:
            return {"error": str(e)}


class StrictJSONDSLParser:
    """
    Parse and validate strict JSON DSL for data queries.
    Schema:
    {
        "intent": "aggregate|groupby|filter|correlation|describe",
        "columns": ["col1", "col2"],
        "filters": [{"column": "col", "operator": "==", "value": x}],
        "groupby": ["col1"],
        "metrics": ["mean", "count"],
        "limit": 1000,
        "explanation": "natural language summary"
    }
    """
    
    VALID_INTENTS = {"aggregate", "groupby", "filter", "correlation", "describe"}
    VALID_OPERATORS = {"==", "!=", ">", "<", ">=", "<=", "in", "not_in"}
    VALID_METRICS = {"mean", "sum", "count", "min", "max", "median", "std"}
    
    @staticmethod
    def validate_dsl(dsl: Dict[str, Any], available_columns: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate DSL structure against schema.
        Returns (is_valid, error_message).
        """
        if not isinstance(dsl, dict):
            return False, "DSL must be a JSON object"
        
        # Check intent
        intent = dsl.get("intent", "describe")
        if intent not in StrictJSONDSLParser.VALID_INTENTS:
            return False, f"Invalid intent: {intent}"
        
        # Check columns exist
        columns = dsl.get("columns", [])
        if not isinstance(columns, list):
            return False, "columns must be a list"
        
        invalid_cols = [c for c in columns if c not in available_columns]
        if invalid_cols:
            return False, f"Unknown columns: {invalid_cols}"
        
        # Check filters
        filters = dsl.get("filters", [])
        if not isinstance(filters, list):
            return False, "filters must be a list"
        
        for filt in filters:
            if not isinstance(filt, dict):
                return False, "Each filter must be an object"
            if "column" not in filt or "operator" not in filt or "value" not in filt:
                return False, "Each filter must have column, operator, value"
            if filt["column"] not in available_columns:
                return False, f"Filter column {filt['column']} not found"
            if filt["operator"] not in StrictJSONDSLParser.VALID_OPERATORS:
                return False, f"Invalid operator: {filt['operator']}"
        
        # Check metrics
        metrics = dsl.get("metrics", [])
        if not isinstance(metrics, list):
            return False, "metrics must be a list"
        
        invalid_metrics = [m for m in metrics if m not in StrictJSONDSLParser.VALID_METRICS]
        if invalid_metrics:
            return False, f"Invalid metrics: {invalid_metrics}"
        
        # Check groupby
        groupby = dsl.get("groupby", [])
        if not isinstance(groupby, list):
            return False, "groupby must be a list"
        
        invalid_groupby = [c for c in groupby if c not in available_columns]
        if invalid_groupby:
            return False, f"Invalid groupby columns: {invalid_groupby}"
        
        return True, None
    
    @staticmethod
    def execute_dsl(dsl: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Execute validated DSL on dataframe"""
        try:
            result_df = df.copy()
            
            # Apply filters
            for filt in dsl.get("filters", []):
                col = filt["column"]
                op = filt["operator"]
                val = filt["value"]
                
                if op == "==":
                    result_df = result_df[result_df[col] == val]
                elif op == "!=":
                    result_df = result_df[result_df[col] != val]
                elif op == ">":
                    result_df = result_df[result_df[col] > val]
                elif op == "<":
                    result_df = result_df[result_df[col] < val]
                elif op == ">=":
                    result_df = result_df[result_df[col] >= val]
                elif op == "<=":
                    result_df = result_df[result_df[col] <= val]
                elif op == "in":
                    result_df = result_df[result_df[col].isin(val)]
                elif op == "not_in":
                    result_df = result_df[~result_df[col].isin(val)]
            
            # Execute based on intent
            intent = dsl.get("intent", "describe")
            
            if intent == "describe":
                return {
                    "intent": "describe",
                    "rows": len(result_df),
                    "columns": list(result_df.columns),
                    "dtypes": result_df.dtypes.astype(str).to_dict(),
                    "sample": result_df.head(3).to_dict('records')
                }
            
            elif intent == "aggregate":
                result = {}
                for col in dsl.get("columns", []):
                    for metric in dsl.get("metrics", []):
                        if metric == "mean" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_mean"] = float(result_df[col].mean())
                        elif metric == "sum" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_sum"] = float(result_df[col].sum())
                        elif metric == "count":
                            result[f"{col}_count"] = int(result_df[col].count())
                        elif metric == "min" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_min"] = float(result_df[col].min())
                        elif metric == "max" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_max"] = float(result_df[col].max())
                        elif metric == "median" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_median"] = float(result_df[col].median())
                        elif metric == "std" and pd.api.types.is_numeric_dtype(result_df[col]):
                            result[f"{col}_std"] = float(result_df[col].std())
                
                return {"intent": "aggregate", "results": result}
            
            elif intent == "groupby":
                groupby_cols = dsl.get("groupby", [])
                if not groupby_cols:
                    return {"error": "groupby requires groupby columns"}
                
                agg_dict = {}
                for col in dsl.get("columns", []):
                    for metric in dsl.get("metrics", ["mean"]):
                        if metric in ["mean", "sum", "min", "max", "median", "std"]:
                            agg_dict[col] = metric
                
                grouped = result_df.groupby(groupby_cols, as_index=False).agg(agg_dict)
                return {
                    "intent": "groupby",
                    "groupby_columns": groupby_cols,
                    "rows": len(grouped),
                    "results": grouped.to_dict('records')[:50]  # Limit results
                }
            
            elif intent == "correlation":
                cols = dsl.get("columns", [])
                numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(result_df[c])]
                
                if len(numeric_cols) < 2:
                    return {"error": "correlation requires at least 2 numeric columns"}
                
                corr = result_df[numeric_cols].corr()
                return {
                    "intent": "correlation",
                    "correlation_matrix": corr.to_dict()
                }
            
            else:
                return {"error": f"Unknown intent: {intent}"}
        
        except Exception as e:
            return {"error": str(e)}


def chat_with_dsl(dataset_id: str, dsl_query: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT JSON DSL INTERFACE (not natural language).
    
    Users must submit JSON matching schema:
    {
        "intent": "aggregate|groupby|filter|correlation|describe",
        "columns": ["col1", "col2"],
        "filters": [...],
        "groupby": [...],
        "metrics": ["mean", "count", ...],
        "limit": 1000
    }
    
    This is SCHEMA-AWARE (not RAG).
    No natural language processing.
    """
    try:
        # Load dataset
        try:
            df = load_clean(dataset_id)
        except:
            df = load_raw(dataset_id)
        
        available_columns = list(df.columns)
        
        # Validate DSL structure
        is_valid, error_msg = StrictJSONDSLParser.validate_dsl(dsl_query, available_columns)
        if not is_valid:
            return {
                "status": "failed",
                "error": error_msg,
                "error_code": "DSL_VALIDATION_FAILED",
                "schema": {
                    "available_columns": available_columns,
                    "valid_intents": list(StrictJSONDSLParser.VALID_INTENTS),
                    "valid_operators": list(StrictJSONDSLParser.VALID_OPERATORS),
                    "valid_metrics": list(StrictJSONDSLParser.VALID_METRICS)
                }
            }
        
        # Execute DSL query
        result = StrictJSONDSLParser.execute_dsl(dsl_query, df)
        
        return {
            "status": "ok",
            "query": dsl_query,
            "result": result
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "DSL_EXECUTION_FAILED"
        }


def get_column_impact(dataset_id: str, column: str) -> Dict[str, Any]:
    """
    Explain how a specific column impacts model predictions.
    Returns evidence-based impact analysis.
    """
    try:
        try:
            df = load_clean(dataset_id)
        except:
            df = load_raw(dataset_id)
        
        if column not in df.columns:
            return {
                "status": "failed",
                "error": f"Column {column} not found",
                "error_code": "COLUMN_NOT_FOUND"
            }
        
        analyzer = ColumnImpactAnalyzer(dataset_id)
        impact_result = analyzer.explain_column_impact(column, df)
        
        return {
            "status": "ok" if "error" not in impact_result else "failed",
            **impact_result
        }
    
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "error_code": "IMPACT_ANALYSIS_FAILED"
        }
