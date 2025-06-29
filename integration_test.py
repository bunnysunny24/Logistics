#!/usr/bin/env python3
"""
Comprehensive Integration Test for Logistics Pulse Copilot
Tests all backend endpoints and validates system functionality
"""
import requests
import json
import time
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, url, data=None, files=None, expected_status=200):
    """Test a single endpoint and report results"""
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        else:
            return f"❌ {name}: Unsupported method {method}"
        
        if response.status_code == expected_status:
            return f"✅ {name}: SUCCESS ({response.status_code})"
        else:
            return f"❌ {name}: FAILED ({response.status_code}) - {response.text[:100]}"
    except Exception as e:
        return f"❌ {name}: ERROR - {str(e)}"

def run_integration_tests():
    """Run comprehensive integration tests"""
    print("🚀 Starting Logistics Pulse Copilot Integration Tests\n")
    
    results = []
    
    # Test 1: Health Check
    results.append(test_endpoint("Health Check", "GET", f"{BASE_URL}/health"))
    
    # Test 2: System Status
    results.append(test_endpoint("System Status", "GET", f"{BASE_URL}/status"))
    
    # Test 3: General Query
    query_data = {
        "query": "What is the current status of shipments?",
        "context": "general"
    }
    results.append(test_endpoint("General Query", "POST", f"{BASE_URL}/query", query_data))
    
    # Test 4: Shipment Query
    shipment_query = {
        "query": "Show me shipment anomalies",
        "context": "shipment"
    }
    results.append(test_endpoint("Shipment Query", "POST", f"{BASE_URL}/query", shipment_query))
    
    # Test 5: Invoice Query
    invoice_query = {
        "query": "Check invoice compliance",
        "context": "invoice"
    }
    results.append(test_endpoint("Invoice Query", "POST", f"{BASE_URL}/query", invoice_query))
    
    # Test 6: Get Anomalies
    results.append(test_endpoint("Get Anomalies", "GET", f"{BASE_URL}/anomalies"))
    
    # Test 7: Trigger Anomaly Detection
    results.append(test_endpoint("Trigger Anomaly Detection", "POST", f"{BASE_URL}/anomalies/detect"))
    
    # Test 8: Get Risk-Based Holds
    results.append(test_endpoint("Get Risk-Based Holds", "GET", f"{BASE_URL}/risk-holds"))
    
    # Test 9: Get Smart Suggestions
    results.append(test_endpoint("Get Smart Suggestions", "GET", f"{BASE_URL}/suggestions"))
    
    # Test 10: File Upload (if test file exists)
    test_file_path = Path("data/invoices/invoice_001.csv")
    if test_file_path.exists():
        with open(test_file_path, 'rb') as f:
            files = {'file': ('invoice_001.csv', f, 'text/csv')}
            data = {'document_type': 'invoice'}
            results.append(test_endpoint("File Upload", "POST", f"{BASE_URL}/upload", data, files))
    else:
        results.append("⚠️ File Upload: SKIPPED - test file not found")
    
    # Test 11: Causal Analysis
    causal_data = {
        "anomaly_id": "test_anomaly_001",
        "context": "shipment delay analysis"
    }
    results.append(test_endpoint("Causal Analysis", "POST", f"{BASE_URL}/causal-analysis", causal_data))
    
    # Test 12: System Statistics
    results.append(test_endpoint("System Statistics", "GET", f"{BASE_URL}/stats"))
    
    # Print Results
    print("\n📊 TEST RESULTS:")
    print("=" * 50)
    for result in results:
        print(result)
    
    # Summary
    passed = len([r for r in results if r.startswith("✅")])
    failed = len([r for r in results if r.startswith("❌")])
    skipped = len([r for r in results if r.startswith("⚠️")])
    
    print(f"\n📈 SUMMARY:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⚠️ Skipped: {skipped}")
    print(f"   📊 Total: {len(results)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! System is ready for demo.")
    else:
        print(f"\n⚠️ {failed} tests failed. Please check the backend logs.")
    
    return passed, failed, skipped

def test_frontend_accessibility():
    """Test if frontend is accessible"""
    try:
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend: Accessible at http://localhost:3000")
            return True
        else:
            print(f"❌ Frontend: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend: Not accessible - {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 LOGISTICS PULSE COPILOT - INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Test backend
    passed, failed, skipped = run_integration_tests()
    
    print("\n" + "=" * 60)
    
    # Test frontend accessibility
    print("\n🌐 FRONTEND ACCESSIBILITY TEST:")
    frontend_ok = test_frontend_accessibility()
    
    print("\n" + "=" * 60)
    print("🎯 FINAL STATUS:")
    
    if failed == 0 and frontend_ok:
        print("🎉 SYSTEM FULLY OPERATIONAL!")
        print("   • Backend: All endpoints working")
        print("   • Frontend: Accessible and integrated")
        print("   • Ready for production demo")
    elif failed == 0:
        print("⚠️ BACKEND OPERATIONAL, FRONTEND ISSUES")
        print("   • Backend: All endpoints working")
        print("   • Frontend: Check if it's running (npm start)")
    else:
        print("❌ SYSTEM ISSUES DETECTED")
        print(f"   • Backend: {failed} endpoint(s) failing")
        print("   • Check backend logs and configuration")
    
    print("\n📋 NEXT STEPS:")
    print("   1. If all tests pass, system is ready for demo")
    print("   2. If tests fail, check backend logs and fix issues")
    print("   3. Start frontend with: cd frontend && npm start")
    print("   4. Access system at: http://localhost:3000")
