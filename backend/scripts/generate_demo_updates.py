import os
import time
from generate_demo_data import (
    DATA_DIR, 
    generate_updated_policy,
    generate_driver_risk_update,
    generate_shipment_anomaly,
    generate_invoice_with_policy_violation
)

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