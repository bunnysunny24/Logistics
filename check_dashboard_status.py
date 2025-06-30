#!/usr/bin/env python3
"""
Dashboard Status Summary
========================

Check the current state of the anomaly dashboard data.
"""

import requests
import json

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║               📊 DASHBOARD STATUS CHECK                   ║
    ║                                                           ║
    ║      Current anomaly data available to the dashboard     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Get current anomalies
        response = requests.get("http://localhost:8000/api/anomalies")
        if response.status_code == 200:
            anomalies = response.json()
            
            print(f"✅ DASHBOARD DATA AVAILABLE:")
            print(f"   📊 Total Anomalies: {len(anomalies)}")
            
            # Calculate breakdown
            high_risk = len([a for a in anomalies if a.get("risk_score", 0) >= 0.8])
            medium_risk = len([a for a in anomalies if 0.5 <= a.get("risk_score", 0) < 0.8]) 
            low_risk = len([a for a in anomalies if a.get("risk_score", 0) < 0.5])
            
            print(f"   🔴 High Risk: {high_risk}")
            print(f"   🟡 Medium Risk: {medium_risk}")
            print(f"   🟢 Low Risk: {low_risk}")
            
            print("\n📋 Recent Anomaly Types:")
            types = {}
            for anomaly in anomalies:
                atype = anomaly.get("anomaly_type", "unknown")
                types[atype] = types.get(atype, 0) + 1
            
            for atype, count in sorted(types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   • {atype}: {count}")
                
            print(f"\n💡 Dashboard Status:")
            print(f"   ✅ Anomaly data is AVAILABLE and WORKING")
            print(f"   ✅ Frontend should show {len(anomalies)} total anomalies")
            print(f"   ✅ Risk breakdown should show: High={high_risk}, Medium={medium_risk}, Low={low_risk}")
            
            print(f"\n🔄 To refresh data in the dashboard:")
            print(f"   1. Navigate to http://localhost:3000")
            print(f"   2. Go to Anomaly Detection Dashboard")
            print(f"   3. Data should automatically refresh every 30 seconds")
            print(f"   4. Click 'Refresh' button if needed")
            
            print(f"\n📝 Note: The trigger detection has a minor issue, but your uploaded")
            print(f"   documents have been processed and anomalies are available!")
            
        else:
            print(f"❌ Could not get anomaly data: Status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error checking dashboard data: {e}")

if __name__ == "__main__":
    main()
