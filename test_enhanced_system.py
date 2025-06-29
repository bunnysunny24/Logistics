#!/usr/bin/env python3
"""
Test script for the enhanced Logistics Pulse Copilot system
Tests RAG model, anomaly detection, and API endpoints
"""

import os
import sys
import json
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import requests
import time

# Add the backend directory to the path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.append(backend_path)

def test_data_loading():
    """Test if data files are available and readable"""
    print("🔍 Testing data loading...")
    
    data_dir = "./data"
    test_results = {
        "invoices": False,
        "shipments": False,
        "policies": False
    }
    
    # Test invoice data
    try:
        invoice_file = f"{data_dir}/invoices/comprehensive_invoices.csv"
        if os.path.exists(invoice_file):
            df = pd.read_csv(invoice_file)
            print(f"  ✅ Loaded {len(df)} invoices from comprehensive_invoices.csv")
            test_results["invoices"] = True
        else:
            print(f"  ❌ Invoice file not found: {invoice_file}")
    except Exception as e:
        print(f"  ❌ Error loading invoice data: {e}")
    
    # Test shipment data
    try:
        shipment_file = f"{data_dir}/shipments/comprehensive_shipments.csv"
        if os.path.exists(shipment_file):
            df = pd.read_csv(shipment_file)
            print(f"  ✅ Loaded {len(df)} shipments from comprehensive_shipments.csv")
            test_results["shipments"] = True
        else:
            print(f"  ❌ Shipment file not found: {shipment_file}")
    except Exception as e:
        print(f"  ❌ Error loading shipment data: {e}")
    
    # Test policy data
    try:
        policy_files = ["payout-rules-v3.md", "shipment-guidelines-v2.md"]
        policy_count = 0
        for policy_file in policy_files:
            file_path = f"{data_dir}/policies/{policy_file}"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                    print(f"  ✅ Loaded policy file: {policy_file} ({len(content)} chars)")
                    policy_count += 1
        
        if policy_count > 0:
            test_results["policies"] = True
    except Exception as e:
        print(f"  ❌ Error loading policy data: {e}")
    
    return test_results

def test_rag_model():
    """Test the RAG model initialization and basic functionality"""
    print("🤖 Testing RAG model...")
    
    try:
        from models.rag_model import LogisticsPulseRAG
        
        # Initialize RAG model
        rag = LogisticsPulseRAG()
        print("  ✅ RAG model initialized successfully")
        
        # Test status
        status = rag.get_status()
        print(f"  📊 RAG Status: {json.dumps(status, indent=2)}")
        
        # Test sample queries
        test_queries = [
            "What are the current invoice anomalies?",
            "Show me shipment delays",
            "What payment terms are violated?"
        ]
        
        for query in test_queries:
            try:
                result = rag.process_query(query)
                print(f"  ✅ Query processed: '{query[:30]}...'")
                print(f"     Confidence: {result.get('confidence', 0):.2f}")
                print(f"     Sources: {len(result.get('sources', []))}")
            except Exception as e:
                print(f"  ❌ Query failed: '{query[:30]}...' - {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Could not import RAG model: {e}")
        return False
    except Exception as e:
        print(f"  ❌ RAG model test failed: {e}")
        return False

def test_anomaly_detector():
    """Test the enhanced anomaly detector"""
    print("🔍 Testing enhanced anomaly detector...")
    
    try:
        from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
        
        # Initialize detector
        detector = EnhancedAnomalyDetector(data_dir="./data")
        print("  ✅ Anomaly detector initialized successfully")
        
        # Test with sample invoice data
        sample_invoice = {
            "invoice_id": "TEST-001",
            "supplier": "ABC Electronics",
            "amount": 25000.0,  # High amount to trigger anomaly
            "currency": "USD",
            "issue_date": "2025-06-29",
            "due_date": "2025-08-29",  # Long payment terms
            "payment_terms": "NET60",
            "early_discount": 0.0,
            "status": "pending",
            "approver": "junior.clerk"  # Insufficient approval level
        }
        
        invoice_anomalies = detector.detect_invoice_anomalies(sample_invoice)
        print(f"  📊 Detected {len(invoice_anomalies)} invoice anomalies")
        
        for anomaly in invoice_anomalies:
            print(f"     - {anomaly.anomaly_type}: {anomaly.description} (Risk: {anomaly.risk_score:.2f})")
        
        # Test with sample shipment data
        sample_shipment = {
            "shipment_id": "TEST-SHP-001",
            "origin": "New York USA",
            "destination": "London UK",
            "carrier": "Suspicious Logistics",  # Should trigger anomaly
            "departure_date": "2025-06-29",
            "estimated_arrival": "2025-07-15",  # Longer than usual
            "actual_arrival": None,
            "status": "In Transit"
        }
        
        shipment_anomalies = detector.detect_shipment_anomalies(sample_shipment)
        print(f"  📊 Detected {len(shipment_anomalies)} shipment anomalies")
        
        for anomaly in shipment_anomalies:
            print(f"     - {anomaly.anomaly_type}: {anomaly.description} (Risk: {anomaly.risk_score:.2f})")
        
        # Test summary
        summary = detector.get_anomalies_summary()
        print(f"  📈 Anomaly summary: {json.dumps(summary, indent=2)}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Could not import anomaly detector: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Anomaly detector test failed: {e}")
        return False

def test_api_endpoints():
    """Test the API endpoints"""
    print("🌐 Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test basic endpoints
    endpoints_to_test = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/api/status", "GET"),
        ("/stats", "GET"),
        ("/api/anomalies", "GET")
    ]
    
    results = {}
    
    for endpoint, method in endpoints_to_test:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
            else:
                response = requests.post(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ {method} {endpoint} - Status: {response.status_code}")
                results[endpoint] = True
            else:
                print(f"  ❌ {method} {endpoint} - Status: {response.status_code}")
                results[endpoint] = False
                
        except requests.exceptions.ConnectionError:
            print(f"  ⚠️ {method} {endpoint} - Server not running")
            results[endpoint] = False
        except Exception as e:
            print(f"  ❌ {method} {endpoint} - Error: {e}")
            results[endpoint] = False
    
    # Test query endpoint
    try:
        query_data = {
            "message": "What are the current invoice anomalies?",
            "context": {}
        }
        response = requests.post(f"{base_url}/api/query", json=query_data, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Query endpoint working - Confidence: {result.get('confidence', 0):.2f}")
            results["/api/query"] = True
        else:
            print(f"  ❌ Query endpoint failed - Status: {response.status_code}")
            results["/api/query"] = False
            
    except Exception as e:
        print(f"  ❌ Query endpoint test failed: {e}")
        results["/api/query"] = False
    
    return results

def test_document_processor():
    """Test document processing capabilities"""
    print("📄 Testing document processor...")
    
    try:
        from utils.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        print("  ✅ Document processor initialized successfully")
        
        # Test with existing invoice files
        data_dir = "./data"
        invoice_files = ["invoice_001.csv", "invoice_002.csv"]
        
        for file_name in invoice_files:
            file_path = f"{data_dir}/invoices/{file_name}"
            if os.path.exists(file_path):
                try:
                    invoice_data = processor.extract_invoice_data(file_path)
                    print(f"  ✅ Processed {file_name}: {invoice_data.get('invoice_id', 'unknown')}")
                except Exception as e:
                    print(f"  ❌ Failed to process {file_name}: {e}")
        
        # Test with existing shipment files
        shipment_files = ["shipment_001.csv", "shipment_002.csv"]
        
        for file_name in shipment_files:
            file_path = f"{data_dir}/shipments/{file_name}"
            if os.path.exists(file_path):
                try:
                    shipment_data = processor.extract_shipment_data(file_path)
                    print(f"  ✅ Processed {file_name}: {shipment_data.get('shipment_id', 'unknown')}")
                except Exception as e:
                    print(f"  ❌ Failed to process {file_name}: {e}")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ Could not import document processor: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Document processor test failed: {e}")
        return False

def run_integration_test():
    """Run a comprehensive integration test"""
    print("🔄 Running integration test...")
    
    try:
        # Test full workflow
        from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
        from utils.document_processor import DocumentProcessor
        
        detector = EnhancedAnomalyDetector(data_dir="./data")
        processor = DocumentProcessor()
        
        # Process a document end-to-end
        data_dir = "./data"
        test_file = f"{data_dir}/invoices/comprehensive_invoices.csv"
        
        if os.path.exists(test_file):
            # Extract data
            df = pd.read_csv(test_file)
            
            total_anomalies = 0
            for _, row in df.head(3).iterrows():  # Test first 3 rows
                invoice_data = row.to_dict()
                anomalies = detector.detect_invoice_anomalies(invoice_data)
                total_anomalies += len(anomalies)
                
                if anomalies:
                    print(f"  📊 Invoice {invoice_data.get('invoice_id')}: {len(anomalies)} anomalies")
            
            print(f"  ✅ Integration test completed: {total_anomalies} total anomalies detected")
            return True
        else:
            print(f"  ❌ Test file not found: {test_file}")
            return False
            
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Logistics Pulse Copilot Enhanced Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run tests
    test_results["data_loading"] = test_data_loading()
    print()
    
    test_results["document_processor"] = test_document_processor()
    print()
    
    test_results["anomaly_detector"] = test_anomaly_detector()
    print()
    
    test_results["rag_model"] = test_rag_model()
    print()
    
    test_results["integration"] = run_integration_test()
    print()
    
    test_results["api_endpoints"] = test_api_endpoints()
    print()
    
    # Summary
    print("=" * 60)
    print("📋 Test Summary:")
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
            if result:
                passed += 1
            total += 1
        elif isinstance(result, dict):
            # For API endpoint results
            endpoint_passed = sum(1 for r in result.values() if r)
            endpoint_total = len(result)
            print(f"  {test_name}: {endpoint_passed}/{endpoint_total} endpoints working")
            passed += endpoint_passed
            total += endpoint_total
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced system is working correctly.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. Some components may need configuration (e.g., OpenAI API key).")
    else:
        print("❌ Several tests failed. Please check the configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
