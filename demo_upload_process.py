#!/usr/bin/env python3
"""
Demo Script: Upload and Process New Data
========================================

This script demonstrates the complete workflow:
1. Upload new data files
2. Trigger anomaly detection
3. Retrieve and display updated anomaly dashboard data

Usage:
    python demo_upload_process.py
"""

import requests
import json
import os
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_DATA_FILE = "test_upload_data.csv"

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          🚀 LOGISTICS PULSE UPLOAD & PROCESS DEMO            ║
    ║                                                               ║
    ║     This demo shows how new data uploads trigger              ║
    ║     automatic anomaly detection and dashboard updates        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
        else:
            print(f"❌ Backend health check failed: Status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Backend is not accessible: {e}")
        return False

def get_current_anomaly_stats():
    """Get current anomaly statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/anomalies")
        if response.status_code == 200:
            anomalies = response.json()
            stats = {
                "total": len(anomalies),
                "high_risk": len([a for a in anomalies if a.get("risk_score", 0) >= 0.8]),
                "medium_risk": len([a for a in anomalies if 0.5 <= a.get("risk_score", 0) < 0.8]),
                "low_risk": len([a for a in anomalies if a.get("risk_score", 0) < 0.5])
            }
            return stats, anomalies
        else:
            print(f"❌ Failed to get anomalies: Status {response.status_code}")
            return None, []
    except Exception as e:
        print(f"❌ Error getting anomalies: {e}")
        return None, []

def upload_test_file():
    """Upload the test data file"""
    if not os.path.exists(TEST_DATA_FILE):
        print(f"❌ Test data file '{TEST_DATA_FILE}' not found")
        return False
    
    try:
        print(f"📤 Uploading test data file: {TEST_DATA_FILE}")
        
        with open(TEST_DATA_FILE, 'rb') as f:
            files = {'file': (TEST_DATA_FILE, f, 'text/csv')}
            response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Upload successful!")
            print(f"   File: {result.get('filename')}")
            print(f"   Size: {result.get('size')} bytes")
            print(f"   Records: {result.get('processing_summary', {}).get('structured_records', 0)}")
            
            anomaly_summary = result.get('processing_summary', {}).get('anomaly_detection', {})
            if anomaly_summary.get('attempted'):
                print(f"   Anomalies detected: {anomaly_summary.get('anomalies_detected', 0)}")
                print(f"     • High Risk: {anomaly_summary.get('high_risk', 0)}")
                print(f"     • Medium Risk: {anomaly_summary.get('medium_risk', 0)}")
                print(f"     • Low Risk: {anomaly_summary.get('low_risk', 0)}")
            
            return True
        else:
            print(f"❌ Upload failed: Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

def trigger_detection():
    """Manually trigger anomaly detection"""
    try:
        print("🔍 Triggering anomaly detection...")
        response = requests.post(f"{API_BASE_URL}/api/detect-anomalies")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Detection completed successfully!")
                print(f"   Total anomalies: {result.get('anomalies', 0)}")
                
                summary = result.get('summary', {})
                if summary:
                    print(f"   • High Risk: {summary.get('high_risk', 0)}")
                    print(f"   • Medium Risk: {summary.get('medium_risk', 0)}")
                    print(f"   • Low Risk: {summary.get('low_risk', 0)}")
                
                return True
            else:
                print(f"❌ Detection failed: {result.get('message')}")
                return False
        else:
            print(f"❌ Detection request failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Detection error: {e}")
        return False

def display_anomaly_details(anomalies, limit=5):
    """Display detailed information about detected anomalies"""
    if not anomalies:
        print("   No anomalies to display")
        return
    
    print(f"\n📊 Anomaly Details (showing first {min(limit, len(anomalies))}):")
    
    for i, anomaly in enumerate(anomalies[:limit]):
        print(f"\n   {i+1}. {anomaly.get('anomaly_type', 'Unknown')}")
        print(f"      Risk Score: {anomaly.get('risk_score', 0):.2f}")
        print(f"      Severity: {anomaly.get('severity', 'unknown')}")
        print(f"      Document: {anomaly.get('document_id', 'unknown')}")
        print(f"      Description: {anomaly.get('description', 'No description')}")
        
        evidence = anomaly.get('evidence', [])
        if evidence:
            print(f"      Evidence: {', '.join(evidence[:2])}")

def main():
    print_banner()
    
    # Step 1: Check backend
    if not check_backend_status():
        print("\n❌ Please start the backend server first:")
        print("   python start_system.py")
        return
    
    # Step 2: Get initial state
    print("\n📊 Getting initial anomaly state...")
    initial_stats, _ = get_current_anomaly_stats()
    if initial_stats:
        print(f"   Current anomalies: {initial_stats['total']}")
        print(f"   • High Risk: {initial_stats['high_risk']}")
        print(f"   • Medium Risk: {initial_stats['medium_risk']}")
        print(f"   • Low Risk: {initial_stats['low_risk']}")
    
    # Step 3: Upload test data
    print("\n🔄 STEP 1: Uploading new test data...")
    if not upload_test_file():
        print("Upload failed. Exiting.")
        return
    
    # Wait a moment for processing
    print("\n⏳ Waiting for processing...")
    time.sleep(2)
    
    # Step 4: Trigger detection (this should now process the uploaded file)
    print("\n🔄 STEP 2: Triggering comprehensive anomaly detection...")
    if not trigger_detection():
        print("Detection failed. Exiting.")
        return
    
    # Wait for detection to complete
    print("\n⏳ Waiting for detection to complete...")
    time.sleep(3)
    
    # Step 5: Get updated state
    print("\n📊 Getting updated anomaly state...")
    final_stats, final_anomalies = get_current_anomaly_stats()
    if final_stats:
        print(f"   Updated anomalies: {final_stats['total']}")
        print(f"   • High Risk: {final_stats['high_risk']}")
        print(f"   • Medium Risk: {final_stats['medium_risk']}")
        print(f"   • Low Risk: {final_stats['low_risk']}")
        
        # Show the change
        if initial_stats:
            change = final_stats['total'] - initial_stats['total']
            if change > 0:
                print(f"   📈 Added {change} new anomalies!")
            elif change < 0:
                print(f"   📉 Reduced by {abs(change)} anomalies")
            else:
                print(f"   ➡️  No change in anomaly count")
    
    # Step 6: Display some anomaly details
    display_anomaly_details(final_anomalies)
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Open the frontend dashboard at http://localhost:3000")
    print("   2. Navigate to the Anomaly Detection Dashboard")
    print("   3. You should see the updated anomaly counts")
    print("   4. Try uploading more files through the UI")
    print("   5. Use the 'Trigger Detection' button to refresh data")

if __name__ == "__main__":
    main()
