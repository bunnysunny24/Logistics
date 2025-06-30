# 🚛 Logistics Pulse Copilot - Real-Time AI Assistant

[![Pathway Hackathon](https://img.shields.io/badge/Pathway-Hackathon-blue)](https://pathway.com/)
[![Real-Time RAG](https://img.shields.io/badge/Real--Time-RAG-green)](https://github.com/pathwaycom/pathway)
[![Track 2](https://img.shields.io/badge/Track-2-orange)](https://github.com/pathwaycom/pathway)

> **🏆 Pathway Hackathon Submission - Track 2: Logistics Pulse Copilot**
> 
> A real-time AI-powered logistics and finance document processing system that detects anomalies, ensures compliance, and provides instant insights using Pathway's streaming ETL pipeline.

## 🎯 Problem Statement

In logistics operations, critical updates happen every few minutes:
- **8:07 AM**: Driver safety status changes from "Low" to "High risk"
- **8:12 AM**: Finance publishes new payout rules with updated rates
- **8:18 AM**: Shipment scan flags "Exception: package missing"

**The Challenge**: If these updates don't surface instantly, bad decisions follow—unsafe drivers stay on the road, wrong rates get quoted, customers wait in the dark.

**Our Solution**: A real-time RAG application that watches live data sources, indexes every new record through Pathway, and proves its currency with instant, up-to-date responses.

## ✨ Hackathon Requirements Compliance

### ✅ **Pathway-Powered Streaming ETL**
- **Core Engine**: Pathway framework handles all data ingestion and processing
- **Real-Time Processing**: Continuously ingests from file directories, APIs, and webhooks
- **Streaming Pipeline**: `backend/pipeline/pathway_ingest.py` implements the backbone ETL

### ✅ **Dynamic Indexing (No Rebuilds)**
- **On-the-Fly Integration**: New data indexed automatically without manual reloads
- **Real-Time Updates**: Data changes flow through to answers immediately
- **No Manual Rebuilds**: Pathway's incremental processing eliminates rebuild needs

### ✅ **Live Retrieval/Generation Interface**
- **Multiple Interfaces**: FastAPI endpoints, React frontend, and direct API access
- **Real-Time Responses**: Answers reflect latest data within seconds
- **Live Updates**: T+0 data changes included in T+1 queries

### ✅ **Demo Video Ready**
- **Before/After Proof**: System designed to showcase live update flow
- **Real-Time Demo**: Add file → trigger update → see new answers immediately
- **Hackathon Validation**: Built-in demo endpoints for judges

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Pathway Engine  │    │  AI Interface   │
│                 │    │                  │    │                 │
│ • CSV Files     │───▶│ • Streaming ETL  │───▶│ • FastAPI       │
│ • PDF Docs      │    │ • Dynamic Index  │    │ • React UI      │
│ • API Feeds     │    │ • Anomaly Engine │    │ • RAG Queries   │
│ • Webhooks      │    │ • Vector Stores  │    │ • Live Updates  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🔄 Real-Time Data Flow

1. **Ingestion**: Pathway monitors `./data/` directories for new files
2. **Processing**: Streaming ETL extracts, transforms, and indexes data
3. **Anomaly Detection**: AI engine flags suspicious patterns in real-time
4. **Vector Indexing**: Documents automatically added to searchable stores
5. **Query Response**: Users get answers reflecting latest data instantly

## ⚡ Getting Started - 30 Second Setup

> **For Hackathon Judges**: Here's the fastest way to see the system in action!

### One-Click Windows Start
```powershell
# Clone and start (all-in-one command)
git clone <repository-url> && cd logistics-pulse-copilot && .\start.ps1
```

### One-Click macOS/Linux Start
```bash
# Clone and start (all-in-one command)
git clone <repository-url> && cd logistics-pulse-copilot && chmod +x start.sh && ./start.sh
```

### Manual Start (3 commands)
```bash
# 1. Setup
pip install -r requirements.txt && python setup_enhanced.py

# 2. Start backend
python backend/main_enhanced.py &

# 3. Test it works
curl http://localhost:8000/api/status
```

**🎯 Verification**: Visit http://localhost:8000/docs to see the live API, or http://localhost:3000 for the frontend UI.

## 🚀 Detailed Setup Guide

### Prerequisites

- **Python 3.8+**
- **Node.js 16+** (for frontend)
- **Git**

### 1. Clone Repository

```bash
git clone https://github.com/your-username/logistics-pulse-copilot.git
cd logistics-pulse-copilot
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env

# Initialize data directories
python setup_enhanced.py
```

### 3. Start the System

#### Option A: Quick Start (Recommended for Demo)
```bash
# Windows
./start.bat

# macOS/Linux
./start.ps1
```

#### Option B: Manual Start
```bash
# Start backend
cd backend
python main_enhanced.py

# Start frontend (new terminal)
cd frontend
npm install
npm start
```

### 4. Verify Installation

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **System Status**: http://localhost:8000/api/status

## 🎮 Demo Instructions

### Real-Time Update Demo

1. **Initial Query**: Ask "What high-risk shipments do we have?"
2. **Add New Data**: Drop a CSV with anomalies into `./data/uploads/`
3. **Watch Magic**: Same query now returns updated results instantly!

### Demo Endpoints

```bash
# Check system status
GET /api/status

# Upload new document
POST /api/upload

# Query with real-time data
POST /api/query

# Trigger anomaly detection
POST /api/detect-anomalies

# Get current anomalies
GET /api/anomalies
```

### Demo Scenarios

#### Scenario 1: Invoice Compliance Alert
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "Are there any non-compliant invoices?"}'
```

#### Scenario 2: Shipment Risk Assessment
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me shipments with route deviations"}'
```

#### Scenario 3: Real-Time Policy Updates
```bash
# Update policy file, then query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the current late fee rates?"}'
```

## 🔧 Key Components

### Pathway Integration

| Component | File | Purpose |
|-----------|------|---------|
| **Streaming ETL** | `backend/pipeline/pathway_ingest.py` | Core Pathway pipeline for data processing |
| **Pipeline Manager** | `backend/pipeline/pathway_manager.py` | Controls and monitors Pathway operations |
| **Real-Time RAG** | `backend/models/rag_model.py` | Integrates Pathway with vector stores |

### RAG System

| Component | File | Purpose |
|-----------|------|---------|
| **Local LLM** | `backend/models/local_llm.py` | Hugging Face model integration |
| **Vector Stores** | `backend/models/rag_model.py` | FAISS + Pathway vector indexing |
| **Anomaly Detection** | `backend/pipeline/enhanced_anomaly_detector.py` | Real-time anomaly flagging |

### API Layer

| Endpoint | Purpose | Real-Time Feature |
|----------|---------|------------------|
| `/api/upload` | Document ingestion | Immediate processing via Pathway |
| `/api/query` | Natural language queries | Latest data always included |
| `/api/anomalies` | Risk alerts | Real-time anomaly detection |
| `/api/status` | System health | Live pipeline monitoring |

## 📊 Use Cases Implemented

### 1. **Driver Safety Monitoring**
- **Real-Time Updates**: Driver risk status changes trigger immediate alerts
- **Example**: "Driver Maya moved from Low to High risk - recommend reassignment"
- **Data Sources**: Safety files, incident reports, performance metrics

### 2. **Invoice & Payment Compliance**
- **Policy Tracking**: System cross-checks invoices against up-to-date contract terms
- **Example**: "Invoice #234 is non-compliant: late-fee clause #4 now applies"
- **Real-Time**: Finance updates → instant policy application

### 3. **Shipment Anomaly & Fraud Detection**
- **Live Monitoring**: Real-time shipment feeds flag suspicious patterns
- **Example**: "Shipment #1027 shows significant route deviation—possible fraud"
- **Instant Investigation**: Pulls relevant policies and historical data immediately

## 🔍 Technical Deep Dive

### Pathway Streaming Architecture

```python
# Core streaming pipeline
class PathwayIngestPipeline:
    def build_pipeline(self):
        # 1. Input connectors for each data type
        invoices = pw.io.fs.read("./data/invoices", format="csv", mode="streaming")
        shipments = pw.io.fs.read("./data/shipments", format="csv", mode="streaming")
        
        # 2. Real-time processing
        processed_docs = self._process_documents(invoices, shipments)
        
        # 3. Anomaly detection
        anomalies = self._detect_anomalies(processed_docs)
        
        # 4. Vector indexing
        self._index_documents(processed_docs)
```

### Dynamic Indexing

```python
# RAG model with Pathway integration
class LogisticsPulseRAG:
    def add_document_to_index(self, content, doc_type, metadata):
        if self.pathway_enabled:
            # Route through Pathway for real-time processing
            self._route_through_pathway(content, doc_type, metadata)
        
        # Also update local store for immediate access
        self._add_to_local_vector_store(content, doc_type, metadata)
```

### Real-Time Query Processing

```python
# Live retrieval with latest data
def process_query(self, query):
    # 1. Sync with Pathway's latest output
    self.sync_with_pathway_index()
    
    # 2. Hybrid retrieval (semantic + keyword)
    docs = self._hybrid_search(query)
    
    # 3. Generate response with fresh data
    return self._generate_response(query, docs)
```

## 📈 Performance & Scalability

- **Latency**: Sub-second query responses with real-time data
- **Throughput**: Handles hundreds of documents per minute
- **Scalability**: Pathway enables horizontal scaling
- **Memory**: Efficient vector store management with FAISS

## 🧪 Testing

### Run Tests
```bash
# Backend tests
python -m pytest backend/tests/

# Integration tests
python test_complete_workflow.py

# Real-time demo
python demo_causal_flow.py
```

### Test Real-Time Updates
```bash
# 1. Start system
python start_system.py

# 2. Upload test data
python test_upload_data.py

# 3. Verify real-time processing
python test_dashboard_update.py
```

## 📋 Data Formats Supported

| Format | Examples | Real-Time Processing |
|--------|----------|---------------------|
| **CSV** | Invoices, shipments, driver data | ✅ Streaming ETL |
| **PDF** | Policies, contracts, reports | ✅ Text extraction + indexing |
| **JSON** | API feeds, webhook data | ✅ Direct processing |
| **Markdown** | Policy documents | ✅ Chunked indexing |

## 🎥 Demo Video Highlights

Our demo video showcases:

1. **Initial State**: System answers query with existing data
2. **Live Update**: New document added to watched directory
3. **Pathway Processing**: Real-time ETL pipeline processes new data
4. **Updated Response**: Same query now includes new information
5. **Proof of Real-Time**: Timestamps show sub-second updates

## 🔮 Future Enhancements

### Agentic RAG (Optional Implementation)
- **LangGraph Integration**: Multi-step reasoning workflows
- **Agent Orchestration**: Intelligent query routing and escalation
- **REST API**: `/api/agents` endpoint for agentic workflows

### Advanced Features
- **Multi-modal Processing**: Images, videos, audio files
- **Webhook Integrations**: Real-time API feeds
- **Advanced Analytics**: Predictive risk modeling
- **Multi-tenant Support**: Enterprise deployment ready

## 📞 Support & Contact

- **Issues**: Open GitHub issues for bugs or questions
- **Discussions**: Use GitHub Discussions for feature requests
- **Documentation**: Check `docs/` folder for detailed guides

## 🏆 Hackathon Submission Checklist

- ✅ **Working Prototype**: Fully functional system with real-time updates
- ✅ **Code Repository**: Complete source code with clear documentation
- ✅ **Pathway Integration**: Core streaming ETL using Pathway framework
- ✅ **Dynamic Indexing**: No manual rebuilds required
- ✅ **Live Interface**: API and UI for real-time queries
- ✅ **Demo Ready**: Built-in demonstration capabilities
- ✅ **Setup Instructions**: Clear installation and running guide

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built for the Pathway Hackathon 2025** 🚀

*Demonstrating the power of real-time RAG for logistics operations*