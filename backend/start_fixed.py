#!/usr/bin/env python3
"""
Simplified startup script for Logistics Pulse Copilot
"""

import uvicorn
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import the app
from main_enhanced import app, rag_model, anomaly_detector, document_processor

def main():
    print("🚀 Starting Logistics Pulse Copilot API v2.0")
    print(f"📊 Data directory: ./data")
    print(f"🔧 Local models only (no external API dependencies)")
    print(f"🤖 Components loaded:")
    print(f"  - RAG Model: {'✅' if rag_model else '❌'}")
    print(f"  - Anomaly Detector: {'✅' if anomaly_detector else '❌'}")
    print(f"  - Document Processor: {'✅' if document_processor else '❌'}")
    
    if not rag_model:
        print("⚠️ RAG Model: Not loaded - queries will use mock responses")
    if not anomaly_detector:
        print("⚠️ Anomaly Detector: Not loaded - anomaly detection disabled")
    if not document_processor:
        print("⚠️ Document Processor: Not loaded - PDF processing disabled")
    
    print("\n🌐 Starting server on http://localhost:8000")
    print("📖 API docs available at http://localhost:8000/docs")
    print("🔄 Use Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()
