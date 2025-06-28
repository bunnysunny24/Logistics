from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
from dotenv import load_dotenv
from loguru import logger

from pipeline.pathway_manager import PathwayManager
from models.rag_model import LogisticsPulseRAG
from utils.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Logistics Pulse Copilot API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
doc_processor = DocumentProcessor()
pathway_manager = PathwayManager()
rag_model = LogisticsPulseRAG()

# Start pathway pipeline in background
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Logistics Pulse Copilot API")
    # Start pathway pipeline in background
    pathway_manager.start_pipeline()
    logger.info("Pathway pipeline started")

class QueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.post("/api/query", response_model=QueryResponse)
async def query_copilot(request: QueryRequest):
    """
    Process a natural language query and return answer with sources
    """
    try:
        response = rag_model.process_query(request.query, request.context)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class DocumentInfo(BaseModel):
    path: str
    type: str  # "invoice", "shipment", "policy"
    metadata: Optional[Dict[str, Any]] = None

@app.post("/api/ingest")
async def ingest_document(doc_info: DocumentInfo, background_tasks: BackgroundTasks):
    """
    Manually trigger ingestion of a specific document
    """
    try:
        background_tasks.add_task(
            pathway_manager.process_document,
            doc_info.path,
            doc_info.type,
            doc_info.metadata
        )
        return {"status": "ingestion_started", "document": doc_info.path}
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/anomalies")
async def get_anomalies(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_risk_score: Optional[float] = 0.5
):
    """
    Get anomalies detected in the system
    """
    try:
        anomalies = pathway_manager.get_anomalies(start_date, end_date, min_risk_score)
        return {"anomalies": anomalies}
    except Exception as e:
        logger.error(f"Error retrieving anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """
    Get system status and statistics
    """
    try:
        # Get status from different components
        pathway_status = pathway_manager.get_status()
        rag_status = rag_model.get_status()
        
        return {
            "status": "operational",
            "pathway": pathway_status,
            "rag": rag_status,
            "documents_processed": pathway_manager.get_document_count(),
            "last_update": pathway_manager.get_last_update_time()
        }
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)