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

app = FastAPI(title="Logistics Pulse API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
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
        "description": "Shipment delayed by 2 days",
        "severity": "medium",
        "timestamp": "2024-01-15T10:30:00Z",
        "details": {
            "shipment_id": "SH001",
            "expected_delivery": "2024-01-15T18:00:00Z",
            "actual_delivery": "2024-01-17T18:00:00Z"
        }
    },
    {
        "id": "2",
        "type": "invoice_discrepancy",
        "description": "Invoice amount mismatch",
        "severity": "high",
        "timestamp": "2024-01-14T14:20:00Z",
        "details": {
            "invoice_id": "INV001",
            "expected_amount": 1500.00,
            "actual_amount": 1800.00
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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Save the uploaded file
        upload_dir = os.path.join("data", "uploads")
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

@app.get("/anomalies")
async def get_anomalies():
    """Get detected anomalies"""
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
    print("Server will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
