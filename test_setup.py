#!/usr/bin/env python3
"""
Test script to verify the Logistics Pulse Copilot setup
"""
import requests
import json
import sys
from datetime import datetime

def test_backend_health():
    """Test if backend is running and healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend not reachable: {e}")
        return False

def test_api_status():
    """Test API status endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ API status endpoint working")
            return True
        else:
            print(f"❌ API status endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API status endpoint not reachable: {e}")
        return False

def test_query_endpoint():
    """Test the query endpoint"""
    try:
        payload = {
            "message": "What is the status of shipments?",
            "context": "test"
        }
        response = requests.post("http://localhost:8000/api/query", 
                                json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "answer" in data and "sources" in data:
                print("✅ Query endpoint working")
                print(f"   Sample response: {data['answer'][:50]}...")
                return True
            else:
                print("❌ Query endpoint returned unexpected format")
                return False
        else:
            print(f"❌ Query endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Query endpoint not reachable: {e}")
        return False

def test_anomalies_endpoint():
    """Test anomalies endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/anomalies", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "anomalies" in data:
                anomalies = data["anomalies"]
                print(f"✅ Anomalies endpoint working ({len(anomalies)} anomalies)")
                
                # Check if anomalies have required fields
                if anomalies:
                    first_anomaly = anomalies[0]
                    required_fields = ['id', 'risk_score', 'anomaly_type', 'document_id']
                    missing_fields = [field for field in required_fields if field not in first_anomaly]
                    
                    if missing_fields:
                        print(f"   ⚠️  Missing fields in anomaly data: {missing_fields}")
                    else:
                        print(f"   ✅ Anomaly data structure is correct")
                        print(f"   📊 Sample: {first_anomaly['anomaly_type']} (risk: {first_anomaly['risk_score']})")
                
                return True
            else:
                print("❌ Anomalies endpoint returned unexpected format")
                return False
        else:
            print(f"❌ Anomalies endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Anomalies endpoint not reachable: {e}")
        return False

def main():
    print("🚀 Testing Logistics Pulse Copilot Setup")
    print("=" * 50)
    
    tests = [
        ("Backend Health", test_backend_health),
        ("API Status", test_api_status),
        ("Query Endpoint", test_query_endpoint),
        ("Anomalies Endpoint", test_anomalies_endpoint),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly.")
        print("\nNext steps:")
        print("1. Start the frontend: cd frontend && npm start")
        print("2. Open http://localhost:3000 in your browser")
        print("3. Upload documents and start chatting!")
    else:
        print("⚠️  Some tests failed. Make sure the backend is running:")
        print("   python -m uvicorn main_simple:app --reload --port 8000")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
