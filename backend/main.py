from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from backend.app import app as omnisearch_app
from backend.config import settings

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("🚀 OmniSearch AI starting...")
    yield
    # Shutdown
    logging.info("🛑 OmniSearch AI shutting down...")

app = FastAPI(
    title="OmniSearch AI 🚀",
    description="Production ML Workbench - EDA + AutoML + Predictions + RAG",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "https://*.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api", omnisearch_app)

@app.get("/")
async def root():
    return {
        "message": "🚀 OmniSearch AI v2.0 - Production Ready ML Workbench",
        "endpoints": ["/api/docs", "/api/redoc"],
        "status": "healthy"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
