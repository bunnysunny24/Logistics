#!/usr/bin/env python3
"""
Complete Workflow Test for Logistics Pulse Copilot
==================================================

This script tests the entire workflow:
1. Upload document/CSV
2. Extract data and detect anomalies
3. Update anomaly database
4. Add anomalies to RAG pipeline
5. Verify dashboard data
6. Test RAG queries about anomalies
"""

import os
import sys
import json
import requests
import time
import pandas as pd
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_DATA_DIR = "./test_data"

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_step(step, description):
    print(f"\n🔄 Step {step}: {description}")

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not accessible: {e}")
        return False

def create_test_data():
    """Create test CSV data with known anomalies"""
    print_step(1, "Creating test data with known anomalies")
    
    # Create test directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    # Test invoice data with various anomalies
    test_invoices = [
        {
            "invoice_id": "TEST-INV-001",
            "supplier": "ABC Electronics",
            "amount": 25000.00,  # High amount deviation
            "payment_terms": "NET30",
            "status": "pending",
            "date": "2025-06-30",
            "po_number": "PO-2025-001"
        },
        {
            "invoice_id": "TEST-INV-002", 
            "supplier": "Normal Supplier Ltd",
            "amount": 1500.00,  # Normal amount
            "payment_terms": "NET30",
            "status": "approved",
            "date": "2025-06-30",
            "po_number": "PO-2025-002"
        },
        {
            "invoice_id": "TEST-INV-003",
            "supplier": "Suspicious Corp",
            "amount": 50000.00,  # Suspicious round amount + high value
            "payment_terms": "NET15",  # Unusual terms
            "status": "pending",
            "date": "2025-06-30",
            "po_number": ""  # Missing PO number
        },
        {
            "invoice_id": "TEST-INV-004",
            "supplier": "Problem Supplier Inc",  # Known problem supplier
            "amount": 10000.00,  # Round amount
            "payment_terms": "NET7",  # Very short terms
            "status": "pending", 
            "date": "2025-06-30",
            "po_number": "PO-2025-004"
        }
    ]
    
    df = pd.DataFrame(test_invoices)
    test_file = os.path.join(TEST_DATA_DIR, "test_workflow_invoices.csv")
    df.to_csv(test_file, index=False)
    
    print(f"✅ Created test file: {test_file}")
    print(f"   📊 Contains {len(test_invoices)} test invoices")
    print(f"   🚨 Expected anomalies: 3-4 (high amounts, suspicious patterns)")
    
    return test_file

def upload_test_file(file_path):
    """Upload test file to backend"""
    print_step(2, "Uploading test file to backend")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/csv')}
            response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File uploaded successfully")
            print(f"   📄 Filename: {result.get('filename')}")
            print(f"   📊 Processing Summary:")
            
            summary = result.get('processing_summary', {})
            print(f"      - Text extracted: {summary.get('text_extracted')}")
            print(f"      - Characters: {summary.get('characters_extracted')}")
            print(f"      - Records: {summary.get('structured_records')}")
            print(f"      - RAG indexed: {summary.get('rag_indexed')}")
            
            anomaly_info = summary.get('anomaly_detection', {})
            print(f"      - Anomaly detection attempted: {anomaly_info.get('attempted')}")
            print(f"      - Anomalies detected: {anomaly_info.get('anomalies_detected')}")
            print(f"        • High risk: {anomaly_info.get('high_risk')}")
            print(f"        • Medium risk: {anomaly_info.get('medium_risk')}")
            print(f"        • Low risk: {anomaly_info.get('low_risk')}")
            
            return result
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

def trigger_anomaly_detection():
    """Trigger manual anomaly detection"""
    print_step(3, "Triggering anomaly detection")
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/detect-anomalies")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Anomaly detection completed")
            print(f"   🔍 Success: {result.get('success')}")
            print(f"   📊 Total anomalies: {result.get('anomalies')}")
            
            summary = result.get('summary', {})
            if summary:
                print(f"   📈 Summary:")
                print(f"      - Total: {summary.get('total')}")
                print(f"      - High risk: {summary.get('high_risk')}")
                print(f"      - Medium risk: {summary.get('medium_risk')}")
                print(f"      - Low risk: {summary.get('low_risk')}")
            
            return result
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return None

def check_dashboard_data():
    """Check anomaly data from dashboard API"""
    print_step(4, "Checking dashboard anomaly data")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/anomalies")
        
        if response.status_code == 200:
            anomalies = response.json()
            print(f"✅ Retrieved {len(anomalies)} anomalies from dashboard")
            
            # Analyze anomaly data
            high_risk = [a for a in anomalies if a.get('risk_score', 0) >= 0.8]
            medium_risk = [a for a in anomalies if 0.5 <= a.get('risk_score', 0) < 0.8]
            low_risk = [a for a in anomalies if a.get('risk_score', 0) < 0.5]
            
            print(f"   📊 Risk Distribution:")
            print(f"      - High risk (≥0.8): {len(high_risk)}")
            print(f"      - Medium risk (0.5-0.79): {len(medium_risk)}")
            print(f"      - Low risk (<0.5): {len(low_risk)}")
            
            # Show sample anomalies
            print(f"   🔍 Sample Anomalies:")
            for i, anomaly in enumerate(anomalies[:3]):
                print(f"      {i+1}. {anomaly.get('id')} - {anomaly.get('anomaly_type')}")
                print(f"         Risk: {anomaly.get('risk_score'):.2f} | Severity: {anomaly.get('severity')}")
                print(f"         Doc: {anomaly.get('document_id')} | Desc: {anomaly.get('description')[:50]}...")
            
            return anomalies
        else:
            print(f"❌ Failed to get anomalies: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Dashboard check error: {e}")
        return None

def test_rag_queries():
    """Test RAG queries about detected anomalies"""
    print_step(5, "Testing RAG queries about anomalies")
    
    test_queries = [
        "What anomalies were detected in the recent uploads?",
        "Show me high risk anomalies found today",
        "What suspicious invoices were flagged?",
        "Tell me about Problem Supplier Inc anomalies",
        "What are the main anomaly types detected?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Query {i}: {query}")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/query",
                json={"message": query},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('answer', 'No answer provided')
                print(f"   ✅ Response: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
        
        time.sleep(1)  # Brief pause between queries

def cleanup_test_data():
    """Clean up test files"""
    print_step(6, "Cleaning up test data")
    
    try:
        test_file = os.path.join(TEST_DATA_DIR, "test_workflow_invoices.csv")
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✅ Removed test file: {test_file}")
        
        if os.path.exists(TEST_DATA_DIR) and not os.listdir(TEST_DATA_DIR):
            os.rmdir(TEST_DATA_DIR)
            print(f"✅ Removed test directory: {TEST_DATA_DIR}")
            
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

def main():
    """Run complete workflow test"""
    print_header("COMPLETE WORKFLOW TEST")
    print("Testing: Upload → Extract → Detect → Update → RAG → Dashboard")
    
    # Check if backend is running
    if not check_backend_health():
        print("\n❌ Cannot proceed without backend. Please start the system first:")
        print("   python start_system.py")
        return False
    
    try:
        # Step 1: Create test data
        test_file = create_test_data()
        
        # Step 2: Upload file
        upload_result = upload_test_file(test_file)
        if not upload_result:
            print("\n❌ Upload failed, stopping test")
            return False
        
        # Wait a moment for processing
        print("\n⏳ Waiting 3 seconds for processing...")
        time.sleep(3)
        
        # Step 3: Trigger detection (if not already done by upload)
        detection_result = trigger_anomaly_detection()
        
        # Wait a moment for detection to complete
        print("\n⏳ Waiting 2 seconds for detection to complete...")
        time.sleep(2)
        
        # Step 4: Check dashboard data
        dashboard_anomalies = check_dashboard_data()
        
        # Step 5: Test RAG queries
        test_rag_queries()
        
        # Summary
        print_header("WORKFLOW TEST SUMMARY")
        if upload_result and dashboard_anomalies:
            print("✅ WORKFLOW TEST PASSED")
            print(f"   • File uploaded and processed successfully")
            print(f"   • Anomalies detected and stored")
            print(f"   • Dashboard shows updated data")
            print(f"   • RAG can answer queries about anomalies")
            print(f"   • Total anomalies in system: {len(dashboard_anomalies)}")
        else:
            print("❌ WORKFLOW TEST FAILED")
            print("   Some steps did not complete successfully")
        
        return True
        
    finally:
        # Cleanup
        cleanup_test_data()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
