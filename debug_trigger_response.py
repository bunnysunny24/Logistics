import requests
import json

try:
    print("Testing trigger detection endpoint...")
    response = requests.post('http://localhost:8000/api/detect-anomalies', timeout=15)
    print(f"Status code: {response.status_code}")
    print(f"Response headers: {dict(response.headers)}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Full response: {json.dumps(result, indent=2)}")
        
        # Check specific fields
        print(f"\nField analysis:")
        print(f"- success: {result.get('success')}")
        print(f"- message: {result.get('message', 'MISSING')}")  
        print(f"- anomalies (new): {result.get('anomalies', 'MISSING')}")
        print(f"- total_anomalies: {result.get('total_anomalies', 'MISSING')}")
        print(f"- summary: {result.get('summary', 'MISSING')}")
        
        if 'summary' in result:
            summary = result['summary']
            print(f"  - summary.total: {summary.get('total', 'MISSING')}")
            print(f"  - summary.new_anomalies: {summary.get('new_anomalies', 'MISSING')}")
            
    else:
        print(f"Error response: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
