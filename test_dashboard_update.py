#!/usr/bin/env python3
"""
Quick Dashboard Update Test
==========================

Test to validate that anomalies are properly being returned to the dashboard.
"""

import requests
import json

API_BASE_URL = "http://localhost:8000"

def test_anomaly_endpoint():
    """Test the anomaly endpoint"""
    print("🔍 Testing anomaly endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/anomalies")
        if response.status_code == 200:
            anomalies = response.json()
            print(f"✅ Anomaly endpoint working - Found {len(anomalies)} anomalies")
            
            if anomalies:
                print("📊 Anomaly breakdown:")
                high_risk = len([a for a in anomalies if a.get("risk_score", 0) >= 0.8])
                medium_risk = len([a for a in anomalies if 0.5 <= a.get("risk_score", 0) < 0.8])
                low_risk = len([a for a in anomalies if a.get("risk_score", 0) < 0.5])
                
                print(f"   • High Risk: {high_risk}")
                print(f"   • Medium Risk: {medium_risk}")
                print(f"   • Low Risk: {low_risk}")
                
                # Show first few anomalies
                print("\n📋 Sample anomalies:")
                for i, anomaly in enumerate(anomalies[:3]):
                    print(f"   {i+1}. {anomaly.get('anomaly_type', 'unknown')} - Risk: {anomaly.get('risk_score', 0):.2f}")
                    print(f"      Description: {anomaly.get('description', 'No description')[:80]}...")
                
                return True
            else:
                print("⚠️ No anomalies found")
                return False
        else:
            print(f"❌ Anomaly endpoint failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing anomaly endpoint: {e}")
        return False

def trigger_detection_test():
    """Test triggering detection"""
    print("\n🔄 Testing anomaly detection trigger...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/api/detect-anomalies")
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Detection successful - Found {result.get('anomalies', 0)} anomalies")
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
            print(f"❌ Detection trigger failed: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing detection trigger: {e}")
        return False

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          🧪 DASHBOARD UPDATE VALIDATION TEST              ║
    ║                                                           ║
    ║     Testing if anomalies are properly returned           ║
    ║     to the frontend dashboard                             ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Check current anomalies
    anomalies_working = test_anomaly_endpoint()
    
    # Test 2: Trigger detection
    detection_working = trigger_detection_test()
    
    # Test 3: Check anomalies again after detection
    print("\n🔄 Re-checking anomalies after detection...")
    anomalies_after = test_anomaly_endpoint()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY:")
    print(f"   ✅ Anomaly Endpoint: {'WORKING' if anomalies_working else 'FAILED'}")
    print(f"   ✅ Detection Trigger: {'WORKING' if detection_working else 'FAILED'}")
    print(f"   ✅ Post-Detection Check: {'WORKING' if anomalies_after else 'FAILED'}")
    
    if all([anomalies_working, detection_working, anomalies_after]):
        print("\n🎉 ALL TESTS PASSED - Dashboard should be showing updated data!")
        print("\n💡 Next steps:")
        print("   1. Refresh your dashboard at http://localhost:3000")
        print("   2. Check the Anomaly Detection section")
        print("   3. You should see the detected anomalies")
    else:
        print("\n⚠️ SOME TESTS FAILED - Check the backend logs for errors")
        print("   - Make sure the backend is running")
        print("   - Check for any error messages in the console")

if __name__ == "__main__":
    main()
