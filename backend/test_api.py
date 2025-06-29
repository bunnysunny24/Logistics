#!/usr/bin/env python3
"""
Simple test script to verify API endpoints are working
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_endpoints():
    """Test API endpoints"""
    print("🧪 Testing API endpoints...\n")
    
    # Test health endpoint
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health endpoint: Working")
        else:
            print(f"❌ Health endpoint: Failed ({response.status_code})")
    except Exception as e:
        print(f"❌ Health endpoint: Error - {e}")
    
    # Test status endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Status endpoint: Working")
            data = response.json()
            print(f"   - Status: {data.get('status', 'unknown')}")
            components = data.get('components', {})
            for comp, status in components.items():
                if isinstance(status, dict):
                    initialized = status.get('initialized', False)
                    print(f"   - {comp}: {'✅' if initialized else '❌'}")
        else:
            print(f"❌ Status endpoint: Failed ({response.status_code})")
    except Exception as e:
        print(f"❌ Status endpoint: Error - {e}")
    
    # Test query endpoint
    try:
        query_data = {"message": "What anomalies have been detected?"}
        response = requests.post(f"{BASE_URL}/api/query", 
                               json=query_data, 
                               headers={"Content-Type": "application/json"},
                               timeout=10)
        if response.status_code == 200:
            print("✅ Query endpoint: Working")
            data = response.json()
            print(f"   - Answer length: {len(data.get('answer', ''))}")
            print(f"   - Confidence: {data.get('confidence', 'N/A')}")
        else:
            print(f"❌ Query endpoint: Failed ({response.status_code})")
    except Exception as e:
        print(f"❌ Query endpoint: Error - {e}")
    
    # Test anomalies endpoint
    try:
        response = requests.get(f"{BASE_URL}/api/anomalies", timeout=5)
        if response.status_code == 200:
            print("✅ Anomalies endpoint: Working")
            data = response.json()
            if isinstance(data, list):
                print(f"   - Found {len(data)} anomalies")
            else:
                print(f"   - Response type: {type(data)}")
        else:
            print(f"❌ Anomalies endpoint: Failed ({response.status_code})")
    except Exception as e:
        print(f"❌ Anomalies endpoint: Error - {e}")

def wait_for_server():
    """Wait for server to be ready"""
    print("⏳ Waiting for server to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        time.sleep(1)
        print(f"   Waiting... ({i+1}/30)")
    
    print("❌ Server not responding after 30 seconds")
    return False

if __name__ == "__main__":
    print("🚀 API Test Script\n")
    
    if wait_for_server():
        test_endpoints()
    else:
        print("Server not available for testing")
    
    print("\n✅ Test complete")
