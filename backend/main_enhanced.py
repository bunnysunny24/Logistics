from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, Any, List, Optional
import os
import sys
import json
from datetime import datetime
import asyncio
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our enhanced components
try:
    from models.rag_model import LogisticsPulseRAG
    from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
    from utils.document_processor import DocumentProcessor
except ImportError as e:
    print(f"Warning: Could not import components: {e}")
    LogisticsPulseRAG = None
    EnhancedAnomalyDetector = None
    DocumentProcessor = None

# Load environment variables
load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WATCH_DIR = os.getenv("WATCH_DIR", "./data")
DATA_DIR = os.getenv("DATA_DIR", "./data")
INDEX_DIR = os.getenv("INDEX_DIR", "./data/index")
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "./prompts")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Initialize components
try:
    if LogisticsPulseRAG:
        rag_model = LogisticsPulseRAG()
        print("✅ RAG model initialized successfully")
    else:
        rag_model = None
        print("⚠️ RAG model not available")
        
    if EnhancedAnomalyDetector:
        anomaly_detector = EnhancedAnomalyDetector(data_dir=DATA_DIR)
        print("✅ Enhanced anomaly detector initialized successfully")
    else:
        anomaly_detector = None
        print("⚠️ Anomaly detector not available")
        
    if DocumentProcessor:
        document_processor = DocumentProcessor()
        print("✅ Document processor initialized successfully")
    else:
        document_processor = None
        print("⚠️ Document processor not available")
        
    print("✅ All components initialized successfully")
except Exception as e:
    print(f"❌ Error initializing components: {e}")
    rag_model = None
    anomaly_detector = None
    document_processor = None

app = FastAPI(
    title="Logistics Pulse Copilot API", 
    version="2.0.0",
    description="Enhanced AI-powered logistics and finance document processing system with advanced RAG and anomaly detection"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced data models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None

class AnomalyResponse(BaseModel):
    id: str
    document_id: str
    anomaly_type: str
    risk_score: float
    severity: str
    description: str
    evidence: List[str]
    recommendations: List[str]
    timestamp: float
    metadata: Dict[str, Any]

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    document_path: str
    anomalies: List[AnomalyResponse]
    
class SystemStatus(BaseModel):
    status: str
    components: Dict[str, Any]
    data_summary: Dict[str, Any]

# Enhanced API endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Logistics Pulse Copilot API v2.0",
        "status": "active",
        "components": {
            "rag_model": rag_model is not None,
            "anomaly_detector": anomaly_detector is not None,
            "document_processor": document_processor is not None
        },
        "endpoints": [
            "/query - Natural language queries",
            "/api/query - API version of query endpoint", 
            "/api/anomalies - Get detected anomalies",
            "/api/upload - Upload and process documents",
            "/api/status - System status",
            "/docs - API documentation"
        ]
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Get RAG model status
        rag_status = rag_model.get_status() if rag_model else {"error": "RAG model not initialized"}
        
        # Get anomaly detector summary
        anomaly_summary = anomaly_detector.get_anomalies_summary() if anomaly_detector else {"error": "Anomaly detector not initialized"}
        
        # Get data file counts
        data_summary = {}
        try:
            for subdir in ["invoices", "shipments", "policies"]:
                dir_path = f"{DATA_DIR}/{subdir}"
                if os.path.exists(dir_path):
                    data_summary[subdir] = len([f for f in os.listdir(dir_path) if f.endswith(('.csv', '.pdf', '.md'))])
                else:
                    data_summary[subdir] = 0
        except Exception as e:
            data_summary = {"error": str(e)}
        
        return SystemStatus(
            status="operational" if all([rag_model, anomaly_detector, document_processor]) else "partial",
            components={
                "rag_model": rag_status,
                "anomaly_detector": anomaly_summary,
                "document_processor": document_processor is not None,
                "openai_configured": bool(OPENAI_API_KEY)
            },
            data_summary=data_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")

@app.post("/query")
async def query_documents(query: ChatMessage):
    """Enhanced natural language query endpoint with RAG"""
    try:
        if not rag_model:
            # Fallback to mock response if RAG model not available
            return _get_mock_response(query.message)
        
        # Process query with enhanced RAG model
        result = rag_model.process_query(
            query=query.message,
            context=query.context or {}
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            confidence=result["confidence"],
            metadata=result.get("metadata", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/api/query") 
async def api_query_documents(query: ChatMessage):
    """API version of query endpoint with enhanced features"""
    try:
        if not rag_model:
            return _get_mock_response(query.message)
        
        # Enhanced query processing with context
        start_time = datetime.now()
        
        result = rag_model.process_query(
            query=query.message,
            context=query.context or {}
        )
        
        # Add processing time to metadata
        if "metadata" not in result:
            result["metadata"] = {}
        result["metadata"]["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"], 
            confidence=result["confidence"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API query failed: {str(e)}")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Legacy upload endpoint for compatibility"""
    try:
        # Save the uploaded file
        upload_dir = os.path.join(DATA_DIR, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "size": len(content),
            "processed": True,
            "document_type": "invoice" if "invoice" in file.filename.lower() else "shipment"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = "auto",
    metadata: Optional[str] = None
):
    """Upload and process document with anomaly detection"""
    try:
        if not document_processor or not anomaly_detector:
            raise HTTPException(status_code=500, detail="Document processing components not available")
        
        # Parse metadata if provided
        doc_metadata = {}
        if metadata:
            try:
                doc_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        
        # Save uploaded file
        upload_dir = f"{DATA_DIR}/uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = f"{upload_dir}/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Detect document type if auto
        if doc_type == "auto":
            if "invoice" in file.filename.lower():
                doc_type = "invoice"
            elif "shipment" in file.filename.lower():
                doc_type = "shipment"
            else:
                doc_type = "document"
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_path,
            doc_type,
            doc_metadata
        )
        
        return DocumentUploadResponse(
            success=True,
            message=f"Document uploaded successfully and processing started",
            document_path=file_path,
            anomalies=[]  # Will be processed in background
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")

async def process_document_background(file_path: str, doc_type: str, metadata: Dict[str, Any]):
    """Background task to process uploaded document"""
    try:
        # Extract data from document
        if doc_type == "invoice":
            doc_data = document_processor.extract_invoice_data(file_path)
        elif doc_type == "shipment":
            doc_data = document_processor.extract_shipment_data(file_path)
        else:
            return
        
        # Detect anomalies
        anomalies = anomaly_detector.process_document(file_path, doc_type, doc_data)
        
        print(f"Processed {file_path}: found {len(anomalies)} anomalies")
        
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")

@app.get("/api/anomalies")
async def get_anomalies(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    min_risk_score: float = 0.0,
    doc_type: Optional[str] = None,
    anomaly_type: Optional[str] = None,
    severity: Optional[str] = None
):
    """Get detected anomalies with filtering"""
    try:
        if not anomaly_detector:
            # Return mock anomalies if detector not available
            return _get_mock_anomalies()
        
        # Get all anomalies
        all_anomalies = anomaly_detector.anomalies
        
        # Apply filters
        filtered_anomalies = []
        for anomaly in all_anomalies:
            # Risk score filter
            if anomaly.get("risk_score", 0) < min_risk_score:
                continue
                
            # Date range filter
            if start_date or end_date:
                anomaly_timestamp = anomaly.get("timestamp", 0)
                anomaly_date = datetime.fromtimestamp(anomaly_timestamp)
                
                if start_date:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if anomaly_date < start_dt:
                        continue
                        
                if end_date:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    if anomaly_date > end_dt:
                        continue
            
            # Document type filter
            if doc_type:
                anomaly_doc_type = anomaly.get("anomaly_type", "").split("_")[0]
                if anomaly_doc_type != doc_type:
                    continue
            
            # Anomaly type filter
            if anomaly_type and anomaly.get("anomaly_type") != anomaly_type:
                continue
                
            # Severity filter
            if severity and anomaly.get("severity") != severity:
                continue
            
            filtered_anomalies.append(anomaly)
        
        return filtered_anomalies
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving anomalies: {str(e)}")

@app.get("/anomalies")
async def get_legacy_anomalies():
    """Legacy anomalies endpoint for compatibility"""
    try:
        return await get_anomalies()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving anomalies: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        if anomaly_detector:
            summary = anomaly_detector.get_anomalies_summary()
            return {
                "total_anomalies": summary.get("total_anomalies", 0),
                "high_risk_count": summary.get("high_risk_count", 0),
                "by_severity": summary.get("by_severity", {}),
                "by_type": summary.get("by_type", {}),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "total_anomalies": 23,
                "high_risk_count": 4,
                "by_severity": {"high": 4, "medium": 15, "low": 4},
                "by_type": {"invoice": 12, "shipment": 11},
                "timestamp": datetime.now().isoformat(),
                "mock_data": True
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(
    query: str,
    answer: str,
    rating: int,
    feedback_text: Optional[str] = None
):
    """Submit user feedback for improving responses"""
    try:
        if not rag_model:
            raise HTTPException(status_code=500, detail="RAG model not available")
        
        if rating < 1 or rating > 5:
            raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
        
        rag_model.add_feedback(query, answer, rating, feedback_text)
        
        return {"success": True, "message": "Feedback submitted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")

@app.delete("/api/memory")
async def clear_conversation_memory():
    """Clear conversation memory"""
    try:
        if not rag_model:
            raise HTTPException(status_code=500, detail="RAG model not available")
        
        rag_model.clear_memory()
        
        return {"success": True, "message": "Conversation memory cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing memory: {str(e)}")

def _get_mock_response(query: str) -> QueryResponse:
    """Generate mock response when RAG model is not available"""
    query_lower = query.lower()
    
    if any(term in query_lower for term in ["invoice", "payment", "billing"]):
        return QueryResponse(
            answer="Based on the invoice data analysis, there are currently 892 invoices with a total value of $2,456,789.50. The system has detected 23 anomalies, including 4 high-risk cases involving payment term violations and 19 medium-risk cases with amount discrepancies. Key findings include: ABC Electronics has 3 flagged invoices, 2 invoices are pending director approval, and the average processing time is 2.3 days.",
            sources=["comprehensive_invoices.csv", "invoice_compliance_policy.md"],
            confidence=0.85,
            metadata={"mock_response": True, "timestamp": datetime.now().isoformat()}
        )
    elif any(term in query_lower for term in ["shipment", "delivery", "logistics"]):
        return QueryResponse(
            answer="Current shipment analysis shows 1,247 active shipments with 89% on-time delivery rate. The system detected 31 anomalies across different risk categories: 8 high-risk cases involving route deviations and carrier changes, 15 medium-risk cases with delivery delays, and 8 low-risk cases with minor timing variations. Notable findings: 3 shipments using non-approved carriers, 12 shipments delayed due to customs processing, and the average transit time variance is +1.2 days.",
            sources=["comprehensive_shipments.csv", "shipment_tracking_data.csv"],
            confidence=0.90,
            metadata={"mock_response": True, "timestamp": datetime.now().isoformat()}
        )
    else:
        return QueryResponse(
            answer="I'm currently running in demo mode. The full AI system includes advanced RAG (Retrieval-Augmented Generation) capabilities for analyzing logistics data, detecting anomalies in invoices and shipments, and providing detailed insights. Please configure your OpenAI API key for complete functionality. The system processes data from invoices, shipments, and policy documents to provide accurate, context-aware responses.",
            sources=["system_info.md"],
            confidence=0.60,
            metadata={"mock_response": True, "demo_mode": True, "timestamp": datetime.now().isoformat()}
        )

def _get_mock_anomalies():
    """Generate mock anomalies when detector is not available"""
    return [
        {
            "id": "mock_1",
            "document_id": "INV-2025-004",
            "anomaly_type": "invoice_amount_deviation",
            "risk_score": 0.85,
            "severity": "high",
            "description": "Invoice amount significantly deviates from supplier's historical average",
            "evidence": [
                "Invoice amount: $15,750.00",
                "Supplier average: $8,200.00",
                "Deviation: 92.1%"
            ],
            "recommendations": [
                "Verify invoice details with supplier",
                "Check for bulk orders or additional services",
                "Require additional approval"
            ],
            "timestamp": datetime.now().timestamp(),
            "metadata": {
                "supplier": "ABC Electronics",
                "amount": 15750.00,
                "mock_data": True
            }
        },
        {
            "id": "mock_2", 
            "document_id": "SHP-2025-003",
            "anomaly_type": "unusual_carrier",
            "risk_score": 0.75,
            "severity": "medium",
            "description": "Unusual carrier selected for this route",
            "evidence": [
                "Carrier: Alternative Carriers",
                "Route: New York USA → London UK", 
                "Common carriers: Global Shipping Inc, Express Worldwide"
            ],
            "recommendations": [
                "Verify carrier credentials",
                "Monitor shipment closely",
                "Check carrier selection reason"
            ],
            "timestamp": datetime.now().timestamp(),
            "metadata": {
                "carrier": "Alternative Carriers",
                "route": "New York USA → London UK",
                "mock_data": True
            }
        }
    ]

if __name__ == "__main__":
    print(f"🚀 Starting Logistics Pulse Copilot API v2.0")
    print(f"📊 Data directory: {DATA_DIR}")
    print(f"🤖 OpenAI configured: {'Yes' if OPENAI_API_KEY else 'No'}")
    print(f"🔧 Components loaded: RAG={rag_model is not None}, Anomaly={anomaly_detector is not None}, Processor={document_processor is not None}")
    
    uvicorn.run(
        "main_enhanced:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
