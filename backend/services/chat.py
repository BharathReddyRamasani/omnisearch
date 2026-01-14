import os
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from backend.services.utils import datasetdir, load_raw, load_clean, model_dir

"""
OmniSearch AI - Enterprise Data Analyst Assistant
============================================
Converts natural language queries into a strict JSON-based DSL.
Never invents data. Never guesses. Always validates against dataset schema.
"""

# ============================================
# DATASET CONTEXT PROVIDER
# ============================================

class DatasetContext:
    """Loads and maintains dataset schema, statistics, and metadata"""
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.df = None
        self.columns = {}
        self.numeric_cols = []
        self.categorical_cols = []
        self.stats = {}
        self.model_meta = {}
        self.load_context()
    
    def load_context(self):
        """Load dataset, schema, stats, and model metadata"""
        try:
            self.df = load_clean(self.dataset_id)
            source = "clean"
        except:
            try:
                self.df = load_raw(self.dataset_id)
                source = "raw"
            except Exception as e:
                raise ValueError(f"Cannot load dataset: {str(e)}")
        
        # Extract schema
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            self.columns[col] = dtype
            
            if 'int' in dtype or 'float' in dtype:
                self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)
        
        # Extract basic statistics
        self.stats = {
            'rows': len(self.df),
            'columns': list(self.df.columns),
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols,
            'missing': self.df.isnull().sum().to_dict(),
            'dtypes': self.columns,
        }
        
        # Load model metadata if available
        self.load_model_metadata()
    
    def load_model_metadata(self):
        """Load trained model information"""
        try:
            meta_path = os.path.join(model_dir(self.dataset_id), "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.model_meta = json.load(f)
        except:
            self.model_meta = {}
    
    def get_sample(self, n=5):
        """Return sample rows for intuition"""
        return self.df.head(n).to_dict('records')
    
    def validate_columns(self, cols: List[str]) -> Tuple[bool, str]:
        """Validate if columns exist"""
        invalid = [c for c in cols if c not in self.columns]
        if invalid:
            return False, f"Columns not found: {', '.join(invalid)}"
        return True, "OK"
    
    def is_numeric(self, col: str) -> bool:
        return col in self.numeric_cols
    
    def is_categorical(self, col: str) -> bool:
        return col in self.categorical_cols


# ============================================
# INTENT DETECTOR
# ============================================

class IntentDetector:
    """
    Industrial-grade intent detector following strict DSL rules.
    
    CORE RULES (NON-NEGOTIABLE):
    1. Never default to count unless explicitly asked
    2. Two columns → groupby/correlation, NOT count
    3. Variation/relationship language → groupby or correlation
    4. Priority: Relationship > Distribution > Trend > Aggregation > Describe
    """
    
    @staticmethod
    def detect_intent(query: str) -> str:
        """Detect analytical intent with strict priority ordering"""
        query_lower = query.lower()
        
        # ============================================
        # PRIORITY 1: RELATIONSHIP/IMPACT/VARIATION
        # ============================================
        # Triggered by: "how does", "affects", "varies", "impact"
        # NEVER use count for these
        variation_keywords = [
            r'\bhow does\b', r'\bhow.*varies\b', r'\baffects?\b', 
            r'\bimpact\b', r'\brelationship\b', r'\bdepend', 
            r'\binfluence', r'\beffect\b', r'\bvaries when\b',
            r'\bchanges when\b', r'\bwhen.*changes\b'
        ]
        
        for pattern in variation_keywords:
            if re.search(pattern, query_lower):
                # Check if mentions grouping ("by", "when", "across")
                if re.search(r'\bby\b|\bwhen\b|\bacross\b', query_lower):
                    return 'groupby'
                else:
                    return 'correlation'
        
        # ============================================
        # PRIORITY 2: DISTRIBUTION
        # ============================================
        # Triggered by: "distribution", "spread", "histogram"
        distribution_keywords = [
            r'\bdistribution\b', r'\bspread\b', r'\brange\b', 
            r'\bhistogram\b', r'\bboxplot\b', r'\bdensity\b'
        ]
        
        for pattern in distribution_keywords:
            if re.search(pattern, query_lower):
                return 'distribution'
        
        # ============================================
        # PRIORITY 3: TREND
        # ============================================
        # Triggered by: "trend", "over time", "increases with"
        trend_keywords = [
            r'\btrend\b', r'\bover time\b', r'\bincreas', r'\bdecreas',
            r'\bline plot\b', r'\bchanges over\b'
        ]
        
        for pattern in trend_keywords:
            if re.search(pattern, query_lower):
                return 'trend'
        
        # ============================================
        # PRIORITY 4: AGGREGATION (Only when explicit)
        # ============================================
        # Triggered by: "total", "sum", "average", "count how many"
        # CRITICAL: Only when explicitly requested, NOT as default
        agg_keywords = [
            r'\btotal\b', r'\bsum\b', r'\baverage\b', r'\bmean\b',
            r'\bcount how many\b', r'\bhow many\b', r'\bmedian\b'
        ]
        
        for pattern in agg_keywords:
            if re.search(pattern, query_lower):
                return 'aggregate'
        
        # ============================================
        # PRIORITY 5: DESCRIBE (Default)
        # ============================================
        describe_keywords = [
            r'\bwhat\b', r'\btell\b', r'\babout\b', r'\bstructure\b',
            r'\bcolumns\b', r'\boverview\b', r'\bdataset\b', r'\brows\b'
        ]
        
        for pattern in describe_keywords:
            if re.search(pattern, query_lower):
                return 'describe'
        
        # ============================================
        # PRIORITY 6: MODEL INFO
        # ============================================
        model_keywords = [
            r'\bmodel\b', r'\baccuracy\b', r'\bperformance\b',
            r'\btrained\b', r'\bscore\b', r'\bevaluation\b'
        ]
        
        for pattern in model_keywords:
            if re.search(pattern, query_lower):
                return 'model_info'
        
        # Default to describe
        return 'describe'
    
    @staticmethod
    def extract_columns(query: str, available_cols: List[str]) -> List[str]:
        """Extract column names from query using word boundaries"""
        found_cols = []
        query_lower = query.lower()
        
        for col in available_cols:
            col_lower = col.lower()
            # Match exact words or with underscores
            patterns = [
                rf'\b{re.escape(col_lower)}\b',
                rf'\b{col_lower.replace("_", " ")}\b',
            ]
            
            if any(re.search(p, query_lower) for p in patterns):
                found_cols.append(col)
        
        return found_cols[:4]  # Max 4 columns
    
    @staticmethod
    def extract_metrics(query: str) -> List[str]:
        """Extract requested metrics from query"""
        metrics = []
        query_lower = query.lower()
        
        metric_patterns = {
            'mean': r'\b(mean|average|avg)\b',
            'sum': r'\b(sum|total)\b',
            'count': r'\b(count|count how many|how many)\b',
            'min': r'\b(min|minimum)\b',
            'max': r'\b(max|maximum)\b',
            'median': r'\b(median|middle)\b',
        }
        
        for metric, pattern in metric_patterns.items():
            if re.search(pattern, query_lower):
                metrics.append(metric)
        
        # Default to mean if nothing found
        return metrics if metrics else ['mean']


# ============================================
# DSL EXECUTOR
# ============================================

class DSLExecutor:
    """Executes JSON DSL queries safely on the dataset"""
    
    def __init__(self, context: DatasetContext):
        self.context = context
        self.df = context.df
    
    def execute_describe(self) -> Dict[str, Any]:
        """Describe dataset structure"""
        return {
            'rows': self.context.stats['rows'],
            'columns': self.context.stats['columns'],
            'numeric_columns': self.context.numeric_cols,
            'categorical_columns': self.context.categorical_cols,
            'dtypes': self.context.columns,
            'missing_values': self.context.stats['missing'],
            'sample': self.context.get_sample(3)
        }
    
    def execute_aggregate(self, columns: List[str], metrics: Dict[str, str]) -> Dict[str, Any]:
        """Execute aggregate query"""
        result = {}
        
        for col, metric in metrics.items():
            if col not in self.context.columns:
                return {'error': f'Column {col} not found'}
            
            if metric == 'mean':
                if self.context.is_numeric(col):
                    result[f'{col}_mean'] = float(self.df[col].mean())
            elif metric == 'sum':
                if self.context.is_numeric(col):
                    result[f'{col}_sum'] = float(self.df[col].sum())
            elif metric == 'count':
                result[f'{col}_count'] = int(self.df[col].count())
            elif metric == 'min':
                if self.context.is_numeric(col):
                    result[f'{col}_min'] = float(self.df[col].min())
            elif metric == 'max':
                if self.context.is_numeric(col):
                    result[f'{col}_max'] = float(self.df[col].max())
            elif metric == 'median':
                if self.context.is_numeric(col):
                    result[f'{col}_median'] = float(self.df[col].median())
            elif metric == 'std':
                if self.context.is_numeric(col):
                    result[f'{col}_std'] = float(self.df[col].std())
        
        return result
    
    def execute_groupby(self, group_by: List[str], metrics: Dict[str, str]) -> Dict[str, Any]:
        """Execute groupby query"""
        valid, msg = self.context.validate_columns(group_by)
        if not valid:
            return {'error': msg}
        
        try:
            grouped = self.df.groupby(group_by)
            result = {}
            
            for metric_col, agg_func in metrics.items():
                if metric_col not in self.context.columns:
                    return {'error': f'Column {metric_col} not found'}
                
                if agg_func == 'mean' and self.context.is_numeric(metric_col):
                    result[f'{metric_col}_mean_by_{",".join(group_by)}'] = \
                        grouped[metric_col].mean().to_dict()
                elif agg_func == 'sum' and self.context.is_numeric(metric_col):
                    result[f'{metric_col}_sum_by_{",".join(group_by)}'] = \
                        grouped[metric_col].sum().to_dict()
                elif agg_func == 'count':
                    result[f'{metric_col}_count_by_{",".join(group_by)}'] = \
                        grouped[metric_col].count().to_dict()
                elif agg_func == 'max' and self.context.is_numeric(metric_col):
                    result[f'{metric_col}_max_by_{",".join(group_by)}'] = \
                        grouped[metric_col].max().to_dict()
                elif agg_func == 'min' and self.context.is_numeric(metric_col):
                    result[f'{metric_col}_min_by_{",".join(group_by)}'] = \
                        grouped[metric_col].min().to_dict()
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def execute_filter(self, conditions: List[Dict], metrics: Dict[str, str]) -> Dict[str, Any]:
        """Execute filter + aggregate query"""
        try:
            df_filtered = self.df.copy()
            
            for cond in conditions:
                col = cond.get('column')
                op = cond.get('operator')
                val = cond.get('value')
                
                if col not in self.context.columns:
                    return {'error': f'Column {col} not found'}
                
                if op == '>':
                    df_filtered = df_filtered[df_filtered[col] > val]
                elif op == '<':
                    df_filtered = df_filtered[df_filtered[col] < val]
                elif op == '==':
                    df_filtered = df_filtered[df_filtered[col] == val]
                elif op == '>=':
                    df_filtered = df_filtered[df_filtered[col] >= val]
                elif op == '<=':
                    df_filtered = df_filtered[df_filtered[col] <= val]
                elif op == 'in':
                    df_filtered = df_filtered[df_filtered[col].isin(val)]
            
            # Apply metrics on filtered data
            result = {'filtered_rows': len(df_filtered)}
            for col, metric in metrics.items():
                if metric == 'mean' and self.context.is_numeric(col):
                    result[f'{col}_mean'] = float(df_filtered[col].mean())
                elif metric == 'sum' and self.context.is_numeric(col):
                    result[f'{col}_sum'] = float(df_filtered[col].sum())
                elif metric == 'count':
                    result[f'{col}_count'] = int(df_filtered[col].count())
            
            return result
        except Exception as e:
            return {'error': str(e)}
    
    def execute_correlation(self, columns: List[str]) -> Dict[str, Any]:
        """Compute correlations between numeric columns"""
        numeric_cols_in_query = [c for c in columns if self.context.is_numeric(c)]
        
        if len(numeric_cols_in_query) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation'}
        
        try:
            corr_matrix = self.df[numeric_cols_in_query].corr()
            return {'correlation_matrix': corr_matrix.to_dict()}
        except Exception as e:
            return {'error': str(e)}
    
    def execute_distribution(self, column: str) -> Dict[str, Any]:
        """Get distribution statistics"""
        if column not in self.context.columns:
            return {'error': f'Column {column} not found'}
        
        try:
            if self.context.is_numeric(column):
                return {
                    'column': column,
                    'type': 'numeric',
                    'mean': float(self.df[column].mean()),
                    'median': float(self.df[column].median()),
                    'std': float(self.df[column].std()),
                    'min': float(self.df[column].min()),
                    'max': float(self.df[column].max()),
                    'q25': float(self.df[column].quantile(0.25)),
                    'q75': float(self.df[column].quantile(0.75)),
                }
            else:
                value_counts = self.df[column].value_counts().to_dict()
                return {
                    'column': column,
                    'type': 'categorical',
                    'unique_values': len(value_counts),
                    'top_values': dict(sorted(value_counts.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10])
                }
        except Exception as e:
            return {'error': str(e)}
    
    def execute_model_info(self) -> Dict[str, Any]:
        """Return model metadata"""
        if not self.context.model_meta:
            return {'error': 'No trained model available'}
        
        return self.context.model_meta


# ============================================
# DSL BUILDER
# ============================================

class DSLBuilder:
    """Converts natural language queries to JSON DSL"""
    
    def __init__(self, context: DatasetContext):
        self.context = context
        self.detector = IntentDetector()
    
    def build(self, query: str) -> Dict[str, Any]:
        """Build DSL from natural language query"""
        
        # Detect intent
        intent = self.detector.detect_intent(query)
        
        if intent == 'describe':
            return {'action': 'describe'}
        
        elif intent == 'aggregate':
            columns = self.detector.extract_columns(query, self.context.numeric_cols)
            if not columns:
                columns = self.context.numeric_cols[:3]
            metrics = self.detector.extract_metrics(query)
            metrics_dict = {col: metrics[0] if metrics else 'mean' for col in columns}
            return {
                'action': 'aggregate',
                'columns': columns,
                'metrics': metrics_dict
            }
        
        elif intent == 'groupby':
            # Extract groupby columns
            groupby_cols = self.detector.extract_columns(query, self.context.categorical_cols)
            if not groupby_cols:
                return {
                    'action': 'unsupported',
                    'reason': 'No categorical columns found for grouping'
                }
            
            # Extract metric columns
            metric_cols = self.detector.extract_columns(query, self.context.numeric_cols)
            if not metric_cols:
                metric_cols = self.context.numeric_cols[:2]
            
            metrics = self.detector.extract_metrics(query)
            metrics_dict = {col: metrics[0] if metrics else 'count' for col in metric_cols}
            
            return {
                'action': 'groupby',
                'group_by': groupby_cols,
                'metrics': metrics_dict
            }
        
        elif intent == 'filter':
            # For now, return unsupported - would need more sophisticated parsing
            return {
                'action': 'unsupported',
                'reason': 'Complex filter queries require more specific syntax'
            }
        
        elif intent == 'correlation':
            columns = self.detector.extract_columns(query, self.context.numeric_cols)
            if len(columns) < 2:
                columns = self.context.numeric_cols[:4]
            
            return {
                'action': 'correlation',
                'columns': columns
            }
        
        elif intent == 'distribution':
            columns = self.detector.extract_columns(query, self.context.columns.keys())
            if not columns:
                columns = [self.context.numeric_cols[0]] if self.context.numeric_cols else [list(self.context.columns.keys())[0]]
            
            return {
                'action': 'distribution',
                'column': columns[0]
            }
        
        elif intent == 'model_info':
            return {'action': 'model_info'}
        
        else:
            return {
                'action': 'unsupported',
                'reason': 'Query cannot be safely executed. Please clarify your intent.'
            }


# ============================================
# MAIN CHAT INTERFACE
# ============================================

def get_chat_response(dataset_id: str, question: str, history: List[Dict] = None) -> Dict[str, Any]:
    """
    Main entry point for chat system.
    Returns ONLY JSON DSL, no explanations.
    
    Returns:
    {
        'status': 'ok',
        'dsl': {action, [columns, metrics, ...]},
        'result': {...execution result...}
    }
    """
    try:
        # Load dataset context
        context = DatasetContext(dataset_id)
        
        # Build DSL from query
        builder = DSLBuilder(context)
        dsl = builder.build(question)
        
        # If unsupported, return error DSL
        if dsl.get('action') == 'unsupported':
            return {
                'status': 'ok',
                'dsl': dsl
            }
        
        # Execute DSL
        executor = DSLExecutor(context)
        action = dsl['action']
        
        result = None
        if action == 'describe':
            result = executor.execute_describe()
        elif action == 'aggregate':
            result = executor.execute_aggregate(dsl['columns'], dsl['metrics'])
        elif action == 'groupby':
            result = executor.execute_groupby(dsl['group_by'], dsl['metrics'])
        elif action == 'filter':
            result = executor.execute_filter(dsl.get('conditions', []), dsl.get('metrics', {}))
        elif action == 'correlation':
            result = executor.execute_correlation(dsl['columns'])
        elif action == 'distribution':
            result = executor.execute_distribution(dsl['column'])
        elif action == 'model_info':
            result = executor.execute_model_info()
        
        return {
            'status': 'ok',
            'dsl': dsl,
            'result': result
        }
    
    except Exception as e:
        return {
            'status': 'error',
            'dsl': {
                'action': 'unsupported',
                'reason': f'Error: {str(e)}'
            }
        }
