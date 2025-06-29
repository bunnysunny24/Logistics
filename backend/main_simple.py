from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WATCH_DIR = os.getenv("WATCH_DIR", "./data")
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "./prompts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(
    title="Logistics Pulse API", 
    version="1.0.0",
    description="AI-powered logistics and finance document processing system"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float

# Mock data for testing
MOCK_ANOMALIES = [
    {
        "id": "1",
        "type": "shipment_delay",
        "anomaly_type": "shipment_delay",
        "description": "Shipment delayed by 2 days",
        "severity": "medium",
        "risk_score": 0.75,
        "timestamp": "2024-01-15T10:30:00Z",
        "document_id": "SH001",
        "details": {
            "shipment_id": "SH001",
            "expected_delivery": "2024-01-15T18:00:00Z",
            "actual_delivery": "2024-01-17T18:00:00Z"
        }
    },
    {
        "id": "2",
        "type": "invoice_discrepancy",
        "anomaly_type": "invoice_discrepancy",
        "description": "Invoice amount mismatch",
        "severity": "high",
        "risk_score": 0.92,
        "timestamp": "2024-01-14T14:20:00Z",
        "document_id": "INV001",
        "details": {
            "invoice_id": "INV001",
            "expected_amount": 1500.00,
            "actual_amount": 1800.00
        }
    },
    {
        "id": "3",
        "type": "delivery_route_anomaly",
        "anomaly_type": "delivery_route_anomaly",
        "description": "Unusual delivery route detected",
        "severity": "low",
        "risk_score": 0.45,
        "timestamp": "2024-01-16T09:15:00Z",
        "document_id": "SH002",
        "details": {
            "shipment_id": "SH002",
            "expected_route": "Route A",
            "actual_route": "Route B",
            "delay_minutes": 45
        }
    },
    {
        "id": "4",
        "type": "payment_timing_anomaly",
        "anomaly_type": "payment_timing_anomaly", 
        "description": "Payment processed outside normal hours",
        "severity": "medium",
        "risk_score": 0.68,
        "timestamp": "2024-01-13T23:45:00Z",
        "document_id": "INV002",
        "details": {
            "invoice_id": "INV002",
            "payment_time": "23:45",
            "normal_hours": "09:00-17:00"
        }
    }
]

MOCK_STATS = {
    "total_shipments": 1247,
    "on_time_deliveries": 1098,
    "delayed_shipments": 149,
    "total_invoices": 892,
    "invoice_amount": 2456789.50,
    "anomalies_detected": 23
}

@app.get("/")
async def root():
    return {"message": "Logistics Pulse API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def api_status_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save the uploaded file
        upload_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Mock processing
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "processed": True,
            "document_type": "invoice" if "invoice" in file.filename.lower() else "shipment"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest and process a document - API version"""
    try:
        # Save the uploaded file
        upload_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Mock processing
        return {
            "success": True,
            "message": "Document ingested successfully",
            "filename": file.filename,
            "size": len(content),
            "processed": True,
            "document_type": "invoice" if "invoice" in file.filename.lower() else "shipment"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {str(e)}")

@app.post("/query")
async def query_documents(query: ChatMessage):
    """Query documents using RAG"""
    try:
        # Mock RAG response
        mock_responses = {
            "shipment": {
                "answer": "Based on the shipment data, there are currently 149 delayed shipments out of 1247 total shipments. The main causes include weather delays and customs processing delays.",
                "sources": ["shipment_001.pdf", "logistics_report.pdf"],
                "confidence": 0.85
            },
            "invoice": {
                "answer": "The invoice analysis shows a total of 892 invoices with a combined amount of $2,456,789.50. There are 23 detected anomalies, primarily involving amount discrepancies.",
                "sources": ["invoice_summary.pdf", "financial_report.pdf"],
                "confidence": 0.92
            },
            "default": {
                "answer": "I found relevant information in the logistics database. The system is tracking shipments, invoices, and detecting anomalies in real-time.",
                "sources": ["general_data.pdf"],
                "confidence": 0.75
            }
        }
        
        # Simple keyword matching for demo
        query_lower = query.message.lower()
        if "shipment" in query_lower or "delivery" in query_lower:
            response = mock_responses["shipment"]
        elif "invoice" in query_lower or "payment" in query_lower:
            response = mock_responses["invoice"]
        else:
            response = mock_responses["default"]
        
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/api/query")
async def api_query_documents(query: ChatMessage):
    """Query documents using RAG - API version"""
    try:
        # Mock RAG response
        mock_responses = {
            "shipment": {
                "answer": "Based on the shipment data, there are currently 149 delayed shipments out of 1247 total shipments. The main causes include weather delays and customs processing delays.",
                "sources": ["shipment_001.pdf", "logistics_report.pdf"],
                "confidence": 0.85
            },
            "invoice": {
                "answer": "The invoice analysis shows a total of 892 invoices with a combined amount of $2,456,789.50. There are 23 detected anomalies, primarily involving amount discrepancies.",
                "sources": ["invoice_summary.pdf", "financial_report.pdf"],
                "confidence": 0.92
            },
            "default": {
                "answer": "I found relevant information in the logistics database. The system is tracking shipments, invoices, and detecting anomalies in real-time.",
                "sources": ["general_data.pdf"],
                "confidence": 0.75
            }
        }
        
        # Simple keyword matching for demo
        query_lower = query.message.lower()
        if "shipment" in query_lower or "delivery" in query_lower:
            response = mock_responses["shipment"]
        elif "invoice" in query_lower or "payment" in query_lower:
            response = mock_responses["invoice"]
        else:
            response = mock_responses["default"]
        
        return QueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/anomalies")
async def get_anomalies():
    """Get detected anomalies"""
    return {"anomalies": MOCK_ANOMALIES, "total": len(MOCK_ANOMALIES)}

@app.get("/api/anomalies")
async def api_get_anomalies():
    """Get detected anomalies - API version"""
    return {"anomalies": MOCK_ANOMALIES, "total": len(MOCK_ANOMALIES)}

@app.get("/stats")
async def get_statistics():
    """Get logistics statistics"""
    return {"stats": MOCK_STATS, "last_updated": datetime.now().isoformat()}

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    # Mock document list
    documents = [
        {
            "id": "doc_001",
            "filename": "shipment_jan_2024.pdf",
            "type": "shipment",
            "uploaded_at": "2024-01-15T10:00:00Z",
            "size": 245678
        },
        {
            "id": "doc_002",
            "filename": "invoice_batch_001.pdf",
            "type": "invoice",
            "uploaded_at": "2024-01-14T15:30:00Z",
            "size": 198432
        }
    ]
    return {"documents": documents, "total": len(documents)}

if __name__ == "__main__":
    print("Starting Logistics Pulse API server...")
    print(f"Server will be available at: http://{HOST}:{PORT}")
    print(f"API docs available at: http://{HOST}:{PORT}/docs")
    print(f"Using OpenAI API: {'✅ Configured' if OPENAI_API_KEY else '❌ Missing'}")
    print(f"Data directory: {DATA_DIR}")
    print(f"LLM Model: {LLM_MODEL}")
    uvicorn.run(app, host=HOST, port=PORT, reload=True)
