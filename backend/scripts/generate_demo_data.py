import os
import csv
import json
import random
from datetime import datetime, timedelta

DATA_DIR = "./data"

def ensure_directories():
    """Ensure all required directories exist"""
    for subdir in ["invoices", "shipments", "policies"]:
        os.makedirs(os.path.join(DATA_DIR, subdir), exist_ok=True)

def generate_initial_invoice():
    """Generate initial invoice that will be affected by later changes"""
    invoice_data = {
        "invoice_id": "INV-2025-001",
        "supplier": "ABC Electronics",
        "amount": 4875.50,
        "currency": "USD",
        "issue_date": "2025-06-15",
        "due_date": "2025-07-15",
        "payment_terms": "NET30",
        "early_discount": 0.02,
        "status": "pending",
        "approver": "john.smith",
        "shipment_id": "SHP-2025-001"
    }
    
    # Write to CSV
    file_path = os.path.join(DATA_DIR, "invoices", "invoice_normal.csv")
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=invoice_data.keys())
        writer.writeheader()
        writer.writerow(invoice_data)
        
    print(f"Generated initial invoice: {file_path}")
    return invoice_data

def generate_initial_shipment():
    """Generate initial shipment that will be affected by later changes"""
    shipment_data = {
        "shipment_id": "SHP-2025-001",
        "origin": "New York, USA",
        "destination": "London, UK",
        "carrier": "Global Shipping Inc",
        "departure_date": "2025-06-15",
        "estimated_arrival": "2025-06-22",
        "actual_arrival": None,
        "status": "In Transit",
        "risk_score": 0.1,
        "anomaly_type": "none",
        "driver_id": "DRV-001"
    }
    
    # Write to CSV
    file_path = os.path.join(DATA_DIR, "shipments", "shipment_normal.csv")
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=shipment_data.keys())
        writer.writeheader()
        writer.writerow(shipment_data)
        
    print(f"Generated initial shipment: {file_path}")
    return shipment_data

def generate_initial_policy():
    """Generate initial payment policy document"""
    policy_text = """# Payment Terms Policy (v2)

## Standard Payment Terms
- NET30 is the default payment term for all suppliers
- Early payment discount: 2% if paid within 10 days

## Overtime Rates
- Standard overtime rate: 1.5x regular hourly rate
- Weekend overtime rate: 2.0x regular hourly rate
- Holiday overtime rate: 2.5x regular hourly rate

## Late Payment Penalties
- 1% fee for payments 1-15 days late
- 2% fee for payments 16-30 days late
- 3% fee for payments over 30 days late

## Approval Requirements
- Invoices under $5,000: Department manager approval
- Invoices $5,000-$10,000: Director approval
- Invoices over $10,000: VP Finance approval
"""
    
    # Write to file
    file_path = os.path.join(DATA_DIR, "policies", "payment_policy_v2.md")
    with open(file_path, 'w') as f:
        f.write(policy_text)
        
    print(f"Generated initial policy: {file_path}")

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
    """Generate all demo data"""
    print("Generating demo data for Logistics Pulse Copilot...")
    ensure_directories()
    
    # Generate initial state data
    generate_initial_invoice()
    generate_initial_shipment()
    generate_initial_policy()
    
    print("\nDemo data generation complete!")
    print("\nTo test the 'before → update → after' scenario:")
    print("1. Start the application and ask: 'What is the status of invoice INV-2025-001?'")
    print("2. Run the following to trigger updates:")
    print("   python scripts/generate_demo_updates.py")
    print("3. Ask the same question again to see the updated response")

if __name__ == "__main__":
    main()