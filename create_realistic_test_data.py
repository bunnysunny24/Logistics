#!/usr/bin/env python3
"""
Create test data with suppliers that match the historical data
This ensures anomaly detection will work properly
"""

import pandas as pd
import os
from datetime import datetime

def create_realistic_test_data():
    """Create test data using existing suppliers from the system"""
    
    # Use suppliers that exist in the enhanced anomaly detector configuration
    test_invoices = [
        {
            "invoice_id": "TEST-INV-001",
            "supplier": "ABC Electronics",  # Known supplier
            "amount": 25000.00,  # Much higher than typical
            "payment_terms": "NET30",
            "status": "pending",
            "issue_date": "2025-06-30",
            "due_date": "2025-07-30",
            "po_number": "PO-2025-001",
            "approver": "John Doe"
        },
        {
            "invoice_id": "TEST-INV-002",
            "supplier": "Problem Supplier Inc",  # Known problem supplier
            "amount": 50000.00,  # Round suspicious amount
            "payment_terms": "NET7",  # Very short terms
            "status": "pending",
            "issue_date": "2025-06-30",
            "due_date": "2025-07-07",
            "po_number": "",  # Missing PO
            "approver": ""  # Missing approver for high amount
        },
        {
            "invoice_id": "TEST-INV-003",
            "supplier": "Tech Solutions Inc",  # Known supplier
            "amount": 10000.00,  # Round amount
            "payment_terms": "NET15",  # Unusual terms for this supplier
            "status": "pending",
            "issue_date": "2025-06-30",
            "due_date": "2025-07-15",
            "po_number": "PO-2025-003",
            "approver": "Jane Smith"
        },
        {
            "invoice_id": "TEST-INV-004",
            "supplier": "Global Logistics",  # Known supplier
            "amount": 99999.99,  # Very suspicious round amount
            "payment_terms": "NET30",
            "status": "pending",
            "issue_date": "2025-06-30",
            "due_date": "2025-07-30", 
            "po_number": "PO-2025-004",
            "approver": ""  # Missing approver for very high amount
        },
        {
            "invoice_id": "TEST-INV-005",
            "supplier": "Acme Corp",  # Known supplier
            "amount": 1500.00,  # Normal amount - should not trigger anomalies
            "payment_terms": "NET30",
            "status": "approved",
            "issue_date": "2025-06-30",
            "due_date": "2025-07-30",
            "po_number": "PO-2025-005",
            "approver": "Bob Johnson"
        }
    ]
    
    df = pd.DataFrame(test_invoices)
    
    # Create test directory
    os.makedirs("./test_data", exist_ok=True)
    
    # Save the test file
    test_file = "./test_data/realistic_test_invoices.csv"
    df.to_csv(test_file, index=False)
    
    print(f"✅ Created realistic test file: {test_file}")
    print(f"📊 Contains {len(test_invoices)} test invoices")
    print(f"🎯 Using known suppliers for better anomaly detection")
    print(f"🚨 Expected anomalies:")
    print(f"   • TEST-INV-001: High amount for ABC Electronics")
    print(f"   • TEST-INV-002: Problem supplier + round amount + missing fields")
    print(f"   • TEST-INV-003: Unusual payment terms")
    print(f"   • TEST-INV-004: Suspicious round amount + missing approver")
    print(f"   • TEST-INV-005: Should be normal (control)")
    
    return test_file

if __name__ == "__main__":
    create_realistic_test_data()
