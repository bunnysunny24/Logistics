import os
import csv
import json
import time
from datetime import datetime

DATA_DIR = "./data"

def generate_updated_policy():
    """Generate updated payment policy with new overtime rates"""
    policy_text = """# Payment Terms Policy (v3)

## Standard Payment Terms
- NET45 is the default payment term for all suppliers (UPDATED)
- Early payment discount: 5% if paid within 7 days (UPDATED)

## Overtime Rates
- Standard overtime rate: 1.75x regular hourly rate (UPDATED)
- Weekend overtime rate: 2.25x regular hourly rate (UPDATED)
- Holiday overtime rate: 3.0x regular hourly rate (UPDATED)

## Late Payment Penalties
- 1.5% fee for payments 1-15 days late (UPDATED)
- 3% fee for payments 16-30 days late (UPDATED)
- 5% fee for payments over 30 days late (UPDATED)

## Approval Requirements
- Invoices under $5,000: Department manager approval
- Invoices $5,000-$10,000: Director approval
- Invoices over $10,000: VP Finance approval
- All rush payments require additional CFO approval (NEW)

## Payment Holds
- All invoices for high-risk drivers are automatically held for review (NEW)
- Shipments with anomalies must be resolved before payment (NEW)
"""
    
    # Write to file
    file_path = os.path.join(DATA_DIR, "policies", "payment_policy_v3.md")
    with open(file_path, 'w') as f:
        f.write(policy_text)
        
    print(f"Generated updated policy: {file_path}")

def generate_driver_risk_update():
    """Generate driver risk update file"""
    driver_data = {
        "driver_id": "DRV-001",
        "name": "Maya Johnson",
        "previous_risk": "Low",
        "current_risk": "High",
        "risk_score": 0.8,
        "update_timestamp": datetime.now().isoformat(),
        "reason": "Three traffic violations in last 30 days"
    }
    
    # Write to JSON
    file_path = os.path.join(DATA_DIR, "driver_risk_update.json")
    with open(file_path, 'w') as f:
        json.dump(driver_data, f, indent=2)
        
    print(f"Generated driver risk update: {file_path}")

def generate_shipment_anomaly():
    """Generate shipment with an anomaly"""
    shipment_data = {
        "shipment_id": "SHP-2025-001",
        "origin": "New York, USA",
        "destination": "London, UK",
        "carrier": "Global Shipping Inc",
        "departure_date": "2025-06-15",
        "estimated_arrival": "2025-06-22",
        "actual_arrival": None,
        "status": "Exception",
        "risk_score": 0.75,
        "anomaly_type": "route_deviation",
        "driver_id": "DRV-001",
        "deviation_kilometers": 250,
        "current_location": "Bristol, UK",
        "expected_location": "London, UK",
        "package_status": "missing"
    }
    
    # Write to CSV
    file_path = os.path.join(DATA_DIR, "shipments", "shipment_anomaly.csv")
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=shipment_data.keys())
        writer.writeheader()
        writer.writerow(shipment_data)
        
    print(f"Generated shipment anomaly: {file_path}")
    return shipment_data

def generate_invoice_with_policy_violation():
    """Generate invoice that violates the new policy"""
    invoice_data = {
        "invoice_id": "INV-2025-001",
        "supplier": "ABC Electronics",
        "amount": 4875.50,
        "currency": "USD",
        "issue_date": "2025-06-15",
        "due_date": "2025-07-15",
        "payment_terms": "NET30",  # Violates new NET45 policy
        "early_discount": 0.02,    # Violates new 5% discount policy
        "status": "on_hold",
        "approver": "john.smith",
        "shipment_id": "SHP-2025-001",
        "hold_reason": "Driver risk and shipment anomaly detected"
    }
    
    # Write to CSV
    file_path = os.path.join(DATA_DIR, "invoices", "invoice_violation.csv")
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=invoice_data.keys())
        writer.writeheader()
        writer.writerow(invoice_data)
        
    print(f"Generated invoice with policy violation: {file_path}")
    return invoice_data

def main():
    """Generate update data to demonstrate 'before → update → after' scenario"""
    print("Generating updates for demo scenario...")
    
    # First update: Policy change
    print("\n[Step 1] Updating payment policy...")
    generate_updated_policy()
    time.sleep(2)  # Allow time for watcher to process
    
    # Second update: Driver risk update
    print("\n[Step 2] Updating driver risk status...")
    generate_driver_risk_update()
    time.sleep(2)  # Allow time for watcher to process
    
    # Third update: Shipment anomaly
    print("\n[Step 3] Generating shipment anomaly...")
    generate_shipment_anomaly()
    time.sleep(2)  # Allow time for watcher to process
    
    # Fourth update: Invoice policy violation and hold
    print("\n[Step 4] Updating invoice with policy violations and hold status...")
    generate_invoice_with_policy_violation()
    
    print("\nUpdates complete! Now ask the system:")
    print("- 'What is the status of invoice INV-2025-001?'")
    print("- 'Why is invoice INV-2025-001 on hold?'")
    print("- 'What's the current overtime rate?'")
    print("- 'What happened to shipment SHP-2025-001?'")

if __name__ == "__main__":
    main()