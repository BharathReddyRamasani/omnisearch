"""
CHAT API ENDPOINTS
==================
FastAPI endpoints for enterprise chat orchestrator.

Endpoints:
POST /chat - Main chat interface
POST /chat/clarification - Handle clarification response
GET /chat/schema/{dataset_id} - Get dataset schema
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.chat.orchestrator import ChatOrchestrator
from backend.services.utils import datasetdir

# Create router
router = APIRouter(prefix="/chat", tags=["chat"])

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ChatRequest(BaseModel):
    """Chat request"""
    dataset_id: str
    query: str
    history: Optional[List[Dict[str, str]]] = None


class ClarificationRequest(BaseModel):
    """Clarification response"""
    dataset_id: str
    choice_index: int
    clarification_context: Dict[str, Any]


class SchemaRequest(BaseModel):
    """Schema request"""
    dataset_id: str


# ============================================
# CACHE FOR ORCHESTRATORS
# ============================================

# Simple in-memory cache (would use Redis in production)
_orchestrators = {}


def get_orchestrator(dataset_id: str) -> ChatOrchestrator:
    """Get or create orchestrator for dataset"""
    
    if dataset_id not in _orchestrators:
        try:
            _orchestrators[dataset_id] = ChatOrchestrator(
                dataset_id=dataset_id,
                use_llm=False,  # Set to True if LLM client is configured
                llm_client=None,
                use_rag=True,
                use_explanation=True
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cannot load dataset: {str(e)}")
    
    return _orchestrators[dataset_id]


# ============================================
# ENDPOINTS
# ============================================

@router.post("/")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    
    Returns:
    {
        "status": "ok|clarification_needed|error",
        "response": "...",
        "dsl": {...},
        "result": {...},
        "confidence": "high|medium|low",
        "audit_id": "..."
    }
    """
    
    try:
        orchestrator = get_orchestrator(request.dataset_id)
        
        response = orchestrator.chat(
            user_query=request.query,
            chat_history=request.history
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clarification")
async def handle_clarification(request: ClarificationRequest):
    """
    Handle clarification response.
    
    User selected one of the clarification options,
    now execute the resolved DSL.
    """
    
    try:
        orchestrator = get_orchestrator(request.dataset_id)
        
        # Get choice text
        options = request.clarification_context.get('options', [])
        if request.choice_index >= len(options):
            raise HTTPException(status_code=400, detail="Invalid choice index")
        
        choice_text = options[request.choice_index].get('interpretation', str(request.choice_index))
        
        # Resolve clarification
        response = orchestrator.resolve_clarification(
            clarification_response=choice_text,
            clarification_context=request.clarification_context
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema/{dataset_id}")
async def get_schema(dataset_id: str):
    """
    Get dataset schema information.
    
    Useful for frontend to understand available columns.
    """
    
    try:
        orchestrator = get_orchestrator(dataset_id)
        
        return {
            'dataset_id': dataset_id,
            'rows': orchestrator.context.stats.get('rows'),
            'columns': orchestrator.context.stats.get('columns'),
            'numeric_columns': orchestrator.context.numeric_cols,
            'categorical_columns': orchestrator.context.categorical_cols,
            'dtypes': orchestrator.context.columns,
            'sample': orchestrator.context.get_sample(3)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/health")
async def health_check():
    """Health check endpoint"""
    return {'status': 'healthy', 'service': 'chat'}
