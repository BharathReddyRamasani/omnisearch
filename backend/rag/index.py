"""
SEMANTIC RAG INDEX
==================
Builds and manages knowledge chunks for semantic understanding.

CRITICAL CONSTRAINT (NON-NEGOTIABLE):
❌ NEVER INCLUDE: Raw data values, PII, row data
✅ ONLY INCLUDE: Column meanings, EDA insights, metadata, business definitions

RAG serves ONE purpose:
→ Help LLM understand DATA MEANING (semantics)
→ NOT data values
→ NOT computation

Chunk Types:
1. Column definitions: What each column represents
2. EDA summaries: Distribution insights
3. ETL metadata: Data transformation history
4. Business context: Domain knowledge
5. Quality metrics: Data completeness, missing patterns
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from backend.services.utils import load_clean, load_raw


@dataclass
class RAGChunk:
    """Single knowledge chunk"""
    id: str
    chunk_type: str  # "column_def", "eda_summary", "etl_metadata", "quality_metric"
    content: str
    dataset_id: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict"""
        return {
            'id': self.id,
            'chunk_type': self.chunk_type,
            'content': self.content,
            'dataset_id': self.dataset_id,
            'metadata': self.metadata
        }


class RAGIndexBuilder:
    """
    Builds knowledge base for semantic understanding.
    
    Safe RAG philosophy:
    - Chunks ONLY contain non-sensitive information
    - Focus on schema and aggregate statistics
    - No raw values, no row data
    """
    
    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self.chunks: List[RAGChunk] = []
        self._load_data()
    
    def _load_data(self):
        """Load dataset for analysis"""
        try:
            try:
                self.df = load_clean(self.dataset_id)
                self.source = 'clean'
            except:
                self.df = load_raw(self.dataset_id)
                self.source = 'raw'
        except Exception as e:
            raise ValueError(f"Cannot load dataset: {str(e)}")
    
    def build_index(self) -> List[RAGChunk]:
        """
        Build complete knowledge index.
        
        Returns:
            List of RAGChunk objects
        """
        self.chunks = []
        
        # Build chunks
        self._create_column_definition_chunks()
        self._create_eda_chunks()
        self._create_quality_chunks()
        self._create_relationship_chunks()
        
        return self.chunks
    
    def _create_column_definition_chunks(self):
        """Create chunks describing each column"""
        
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            
            # Basic stats
            non_null = self.df[col].notna().sum()
            null_pct = (1 - non_null / len(self.df)) * 100
            
            # Type-specific info
            type_str = self._infer_type_meaning(col, dtype)
            
            if 'int' in dtype or 'float' in dtype:
                # Numeric column
                mean = self.df[col].mean()
                std = self.df[col].std()
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                
                content = (
                    f"Column '{col}' is numeric ({dtype}).\n"
                    f"Type: {type_str}\n"
                    f"Coverage: {non_null}/{len(self.df)} ({100-null_pct:.1f}%)\n"
                    f"Range: [{min_val:.2f}, {max_val:.2f}]\n"
                    f"Statistics: mean={mean:.2f}, std={std:.2f}"
                )
            else:
                # Categorical column
                unique = self.df[col].nunique()
                top_val = self.df[col].value_counts().index[0] if len(self.df) > 0 else None
                
                content = (
                    f"Column '{col}' is categorical ({dtype}).\n"
                    f"Type: {type_str}\n"
                    f"Coverage: {non_null}/{len(self.df)} ({100-null_pct:.1f}%)\n"
                    f"Unique values: {unique}\n"
                    f"Most common: {top_val}"
                )
            
            chunk = RAGChunk(
                id=f"col_def_{col}",
                chunk_type="column_def",
                content=content,
                dataset_id=self.dataset_id,
                metadata={
                    'column': col,
                    'dtype': dtype,
                    'null_pct': round(null_pct, 2)
                }
            )
            self.chunks.append(chunk)
    
    def _create_eda_chunks(self):
        """Create chunks with EDA insights"""
        
        # CHUNK 1: Dataset overview
        content = (
            f"Dataset '{self.dataset_id}' overview:\n"
            f"Rows: {len(self.df)}\n"
            f"Columns: {len(self.df.columns)}\n"
            f"Data source: {self.source}\n"
            f"Memory usage: ~{self.df.memory_usage(deep=True).sum() / 1e6:.1f} MB"
        )
        
        chunk = RAGChunk(
            id="eda_overview",
            chunk_type="eda_summary",
            content=content,
            dataset_id=self.dataset_id,
            metadata={'type': 'dataset_overview'}
        )
        self.chunks.append(chunk)
        
        # CHUNK 2: Missing data patterns
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_cols = missing[missing > 0].sort_values(ascending=False)
            content = "Missing data patterns:\n"
            for col, count in missing_cols.items():
                pct = (count / len(self.df)) * 100
                content += f"  {col}: {count} ({pct:.1f}%)\n"
        else:
            content = "No missing data detected. Dataset is complete."
        
        chunk = RAGChunk(
            id="eda_missing",
            chunk_type="eda_summary",
            content=content,
            dataset_id=self.dataset_id,
            metadata={'type': 'missing_patterns'}
        )
        self.chunks.append(chunk)
        
        # CHUNK 3: Numeric columns summary
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            content = f"Numeric columns ({len(numeric_cols)}):\n"
            for col in numeric_cols[:10]:  # Max 10
                content += f"  - {col}\n"
            
            chunk = RAGChunk(
                id="eda_numeric",
                chunk_type="eda_summary",
                content=content,
                dataset_id=self.dataset_id,
                metadata={'type': 'numeric_summary'}
            )
            self.chunks.append(chunk)
        
        # CHUNK 4: Categorical columns summary
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            content = f"Categorical columns ({len(cat_cols)}):\n"
            for col in cat_cols[:10]:  # Max 10
                unique = self.df[col].nunique()
                content += f"  - {col}: {unique} unique values\n"
            
            chunk = RAGChunk(
                id="eda_categorical",
                chunk_type="eda_summary",
                content=content,
                dataset_id=self.dataset_id,
                metadata={'type': 'categorical_summary'}
            )
            self.chunks.append(chunk)
    
    def _create_quality_chunks(self):
        """Create chunks with data quality metrics"""
        
        completeness = (self.df.notna().sum() / len(self.df) * 100).mean()
        
        content = (
            f"Data quality metrics:\n"
            f"Overall completeness: {completeness:.1f}%\n"
            f"Duplicate rows: {self.df.duplicated().sum()}\n"
            f"Data types represented: {self.df.dtypes.nunique()}"
        )
        
        chunk = RAGChunk(
            id="quality_metrics",
            chunk_type="quality_metric",
            content=content,
            dataset_id=self.dataset_id,
            metadata={'type': 'quality_overview'}
        )
        self.chunks.append(chunk)
    
    def _create_relationship_chunks(self):
        """Create chunks about column relationships"""
        
        # Find numeric columns for correlation
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            
            # Find strong correlations
            strong_corrs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    corr = corr_matrix.loc[col1, col2]
                    if abs(corr) > 0.7:
                        strong_corrs.append((col1, col2, corr))
            
            if strong_corrs:
                strong_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
                content = "Strong correlations detected:\n"
                for col1, col2, corr in strong_corrs[:5]:
                    content += f"  {col1} ↔ {col2}: {corr:.3f}\n"
                
                chunk = RAGChunk(
                    id="relationships_strong_corr",
                    chunk_type="eda_summary",
                    content=content,
                    dataset_id=self.dataset_id,
                    metadata={'type': 'strong_correlations'}
                )
                self.chunks.append(chunk)
    
    def _infer_type_meaning(self, col: str, dtype: str) -> str:
        """Infer what a column represents based on name and type"""
        
        col_lower = col.lower()
        
        # Heuristics
        if any(x in col_lower for x in ['id', 'uuid', 'pk']):
            return 'Identifier'
        if any(x in col_lower for x in ['date', 'time', 'created', 'updated']):
            return 'Timestamp'
        if any(x in col_lower for x in ['price', 'cost', 'amount', 'salary', 'revenue']):
            return 'Financial metric'
        if any(x in col_lower for x in ['count', 'quantity', 'num']):
            return 'Count/Quantity'
        if any(x in col_lower for x in ['percent', 'pct', 'ratio']):
            return 'Percentage/Ratio'
        if any(x in col_lower for x in ['category', 'type', 'class', 'group']):
            return 'Category'
        if any(x in col_lower for x in ['region', 'location', 'area', 'zone']):
            return 'Geographic'
        
        return 'Generic'


class RAGRetriever:
    """
    Retrieves relevant chunks for semantic understanding.
    
    WARNING: Used ONLY to help LLM understand schema/meaning.
    Output is sent to LLM prompt ONLY as context.
    LLM still cannot hallucinate - DSL validation gates all output.
    """
    
    def __init__(self, chunks: List[RAGChunk]):
        self.chunks = chunks
    
    def retrieve_for_query(self, query: str, k: int = 5) -> List[RAGChunk]:
        """
        Simple keyword-based retrieval (no embeddings needed for MVP).
        
        Args:
            query: User query
            k: Number of chunks to retrieve
        
        Returns:
            Top k relevant chunks
        """
        
        # Extract keywords from query
        query_words = set(query.lower().split())
        
        # Score chunks
        scores = []
        for chunk in self.chunks:
            # Score based on keyword overlap
            chunk_words = set(chunk.content.lower().split())
            overlap = len(query_words & chunk_words)
            
            # Boost column definitions
            if chunk.chunk_type == 'column_def':
                overlap *= 1.5
            
            scores.append((chunk, overlap))
        
        # Sort and return top k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scores[:k]]
    
    def get_context_for_lm(self, chunks: List[RAGChunk]) -> str:
        """
        Format chunks as context for LLM prompt.
        
        ⚠️ This context is informational ONLY.
        LLM still validates against DSL schema and dataset schema.
        """
        
        if not chunks:
            return ""
        
        context_lines = ["KNOWLEDGE CONTEXT:"]
        for chunk in chunks:
            context_lines.append(f"\n[{chunk.chunk_type}]")
            context_lines.append(chunk.content)
        
        return "\n".join(context_lines)
