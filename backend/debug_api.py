#!/usr/bin/env python3
"""
Debug script to see actual error responses from API endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_status_endpoint():
    """Test status endpoint and show detailed error"""
    print("🔍 Testing /api/status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/status", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Status endpoint working!")
            print(json.dumps(data, indent=2))
        else:
            print("❌ Status endpoint failed")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_anomalies_endpoint():
    """Test anomalies endpoint and show detailed error"""
    print("\n🔍 Testing /api/anomalies endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/anomalies", timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Anomalies endpoint working!")
            print(f"Found {len(data)} anomalies")
        else:
            print("❌ Anomalies endpoint failed")
    except Exception as e:
        print(f"❌ Request failed: {e}")

def test_query_endpoint():
    """Test query endpoint"""
    print("\n🔍 Testing /api/query endpoint...")
    try:
        query_data = {"message": "Show me anomaly information"}
        response = requests.post(f"{BASE_URL}/api/query", 
                               json=query_data, 
                               headers={"Content-Type": "application/json"},
                               timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Query endpoint working!")
            print(f"Answer preview: {data.get('answer', '')[:100]}...")
        else:
            print("❌ Query endpoint failed")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("🐛 API Debug Test\n")
    test_status_endpoint()
    test_anomalies_endpoint()
    test_query_endpoint()
