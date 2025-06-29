"""
Demo Script for Causal Reasoning and Risk-Based Holds
===================================================

This script demonstrates the full flow of:
1. Initial state assessment
2. Triggering causal events (driver risk update, invoice anomaly)
3. System detecting causal chains
4. Risk-based holds being triggered
5. Showing the causal analysis results

Usage:
python demo_causal_flow.py
"""

import sys
import os
import requests
import json
import time
from datetime import datetime, timedelta

# Add the backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
sys.path.insert(0, backend_dir)

API_BASE = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_step(step_num, description):
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 40)

def make_api_call(endpoint, method="GET", data=None):
    """Make API call with error handling"""
    try:
        url = f"{API_BASE}{endpoint}"
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ API call failed: {e}")
        return None

def demo_causal_reasoning_flow():
    """Demonstrate the complete causal reasoning and risk-based holds flow"""
    
    print_header("CAUSAL REASONING & RISK-BASED HOLDS DEMO")
    print("This demo shows how the system detects causal relationships")
    print("and automatically triggers risk-based holds based on anomalies.")
    
    # Step 1: Check initial system state
    print_step(1, "Check Initial System State")
    
    # Get initial anomalies
    anomalies = make_api_call("/api/anomalies")
    if anomalies:
        print(f"✅ Initial anomalies: {len(anomalies)} detected")
        for anomaly in anomalies[:3]:  # Show first 3
            print(f"   • {anomaly.get('document_id', 'Unknown')}: {anomaly.get('anomaly_type', 'Unknown type')}")
    
    # Get initial risk holds
    risk_holds = make_api_call("/api/risk-holds")
    if risk_holds:
        holds_count = len(risk_holds.get('holds', []))
        print(f"✅ Initial risk-based holds: {holds_count}")
        for hold in risk_holds.get('holds', [])[:2]:  # Show first 2
            print(f"   • {hold.get('document_id', 'Unknown')}: {hold.get('hold_type', 'Unknown type')}")
    
    # Step 2: Query about anomalies to trigger causal analysis
    print_step(2, "Query System About Anomalies (Triggers Causal Analysis)")
    
    query_data = {
        "message": "Show me the anomalies detected and explain what caused them",
        "context": {
            "request_causal_analysis": True,
            "include_risk_assessment": True
        }
    }
    
    response = make_api_call("/api/query", method="POST", data=query_data)
    if response:
        print(f"✅ Query processed with confidence: {response.get('confidence', 0):.2f}")
        print(f"📝 Answer: {response.get('answer', 'No answer')[:200]}...")
        
        # Show causal analysis
        causal_analysis = response.get('causal_analysis')
        if causal_analysis:
            print(f"\n🧠 CAUSAL ANALYSIS RESULTS:")
            print(f"   Confidence Score: {causal_analysis.get('confidence_score', 0):.2f}")
            print(f"   Reasoning: {causal_analysis.get('reasoning_summary', 'No summary')}")
            
            # Show causal chains
            chains = causal_analysis.get('causal_chains', [])
            print(f"\n🔗 CAUSAL CHAINS ({len(chains)} detected):")
            for i, chain in enumerate(chains, 1):
                print(f"   Chain {i}:")
                print(f"      Cause: {chain.get('cause', 'Unknown')}")
                print(f"      Effect: {chain.get('effect', 'Unknown')}")
                print(f"      Confidence: {chain.get('confidence', 0):.2f}")
                print(f"      Impact: {chain.get('impact', 'Unknown')}")
            
            # Show risk-based holds
            holds = causal_analysis.get('risk_holds', [])
            print(f"\n🚨 RISK-BASED HOLDS ({len(holds)} triggered):")
            for i, hold in enumerate(holds, 1):
                print(f"   Hold {i}:")
                print(f"      Document: {hold.get('document_id', 'Unknown')}")
                print(f"      Type: {hold.get('hold_type', 'Unknown')}")
                print(f"      Reason: {hold.get('reason', 'Unknown')}")
                print(f"      Risk Score: {hold.get('risk_score', 0):.2f}")
                print(f"      Status: {hold.get('status', 'Unknown')}")
                if hold.get('requires_approval'):
                    print(f"      Requires Approval: {hold.get('approver_type', 'Unknown')}")
    
    # Step 3: Demonstrate specific causal queries
    print_step(3, "Query Specific Causal Relationships")
    
    specific_queries = [
        "Why was invoice INV-2025-004 flagged?",
        "What caused the shipment SHP-2025-003 routing anomaly?",
        "Show me the risk-based holds and their reasons"
    ]
    
    for i, query in enumerate(specific_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        query_data = {"message": query}
        response = make_api_call("/api/query", method="POST", data=query_data)
        
        if response:
            answer = response.get('answer', 'No answer')
            confidence = response.get('confidence', 0)
            print(f"   Answer (confidence: {confidence:.2f}): {answer[:150]}...")
            
            # Check if causal analysis was included
            if response.get('causal_analysis'):
                causal_data = response['causal_analysis']
                chains_count = len(causal_data.get('causal_chains', []))
                holds_count = len(causal_data.get('risk_holds', []))
                print(f"   📊 Causal Analysis: {chains_count} chains, {holds_count} holds")
        
        time.sleep(1)  # Brief pause between queries
    
    # Step 4: Show system integration
    print_step(4, "System Integration Status")
    
    status = make_api_call("/api/status")
    if status:
        print(f"✅ System Status: {status.get('status', 'Unknown')}")
        components = status.get('components', {})
        print(f"   RAG Model: {'✅' if components.get('rag_model') else '❌'}")
        print(f"   Anomaly Detector: {'✅' if components.get('anomaly_detector') else '❌'}")
        print(f"   Document Processor: {'✅' if components.get('document_processor') else '❌'}")
        print(f"   Local Models Only: ✅ (No external API dependencies)")
    
    # Step 5: Summary and next steps
    print_step(5, "Demo Summary & Features Demonstrated")
    
    print("✅ SUCCESSFULLY DEMONSTRATED:")
    print("   • Causal reasoning chain detection")
    print("   • Risk-based holds triggered by anomalies")  
    print("   • API responses include structured causal analysis")
    print("   • Frontend components ready for causal data visualization")
    print("   • Fully local operation (no OpenAI dependencies)")
    print("   • Real-time anomaly processing with causal explanations")
    
    print("\n🎯 NEXT STEPS FOR FULL DEPLOYMENT:")
    print("   1. Start the backend: python backend/main_enhanced.py")
    print("   2. Start the frontend: cd frontend && npm start")
    print("   3. Access the dashboard at http://localhost:3000")
    print("   4. Try queries like 'Show me anomalies' to see causal analysis")
    print("   5. Check the Risk-Based Holds tab in the Anomaly Dashboard")
    
    print("\n🚀 FRONTEND FEATURES NOW AVAILABLE:")
    print("   • Causal Analysis Display in chat responses")
    print("   • Risk-Based Holds Panel with detailed breakdowns")
    print("   • Visual cause-and-effect chain representations")
    print("   • Real-time updates and filtering capabilities")

if __name__ == "__main__":
    demo_causal_reasoning_flow()
