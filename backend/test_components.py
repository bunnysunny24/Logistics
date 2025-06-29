#!/usr/bin/env python3
"""
Quick test script to check component availability and basic functionality
"""

import os
import sys
import traceback

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing component imports...")
    
    # Test RAG model
    try:
        from models.rag_model import LogisticsPulseRAG
        print("✅ RAG model import: Success")
        rag_available = True
    except Exception as e:
        print(f"❌ RAG model import: Failed - {e}")
        rag_available = False
    
    # Test Anomaly Detector
    try:
        from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
        print("✅ Anomaly detector import: Success")
        anomaly_available = True
    except Exception as e:
        print(f"❌ Anomaly detector import: Failed - {e}")
        anomaly_available = False
    
    # Test Document Processor
    try:
        from utils.document_processor import PDFProcessor
        print("✅ Document processor import: Success")
        processor_available = True
    except Exception as e:
        print(f"❌ Document processor import: Failed - {e}")
        processor_available = False
    
    return rag_available, anomaly_available, processor_available

def test_component_initialization():
    """Test component initialization"""
    print("\n🔧 Testing component initialization...")
    
    rag_available, anomaly_available, processor_available = test_imports()
    
    # Test RAG initialization
    if rag_available:
        try:
            from models.rag_model import LogisticsPulseRAG
            rag = LogisticsPulseRAG()
            print("✅ RAG model initialization: Success")
        except Exception as e:
            print(f"❌ RAG model initialization: Failed - {e}")
    
    # Test Anomaly Detector initialization
    if anomaly_available:
        try:
            from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
            detector = EnhancedAnomalyDetector(data_dir="./data")
            print("✅ Anomaly detector initialization: Success")
        except Exception as e:
            print(f"❌ Anomaly detector initialization: Failed - {e}")
    
    # Test Document Processor initialization
    if processor_available:
        try:
            from utils.document_processor import PDFProcessor
            processor = PDFProcessor()
            print("✅ Document processor initialization: Success")
        except Exception as e:
            print(f"❌ Document processor initialization: Failed - {e}")

def test_data_directories():
    """Test if data directories exist"""
    print("\n📁 Testing data directories...")
    
    required_dirs = [
        "./data",
        "./data/invoices",
        "./data/shipments",
        "./data/policies",
        "./data/uploads"
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"✅ {dir_path}: Exists ({file_count} files)")
        else:
            print(f"❌ {dir_path}: Missing")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✅ Created directory: {dir_path}")
            except Exception as e:
                print(f"  ❌ Failed to create directory: {e}")

def test_dependencies():
    """Test if key dependencies are available"""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        'fastapi',
        'uvicorn',
        'pandas',
        'numpy',
        'pydantic',
        'python-dotenv',
        'watchdog',
        'loguru'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}: Available")
        except ImportError:
            print(f"❌ {dep}: Missing - install with 'pip install {dep}'")

def create_mock_files():
    """Create mock data files if they don't exist"""
    print("\n📄 Creating mock data files...")
    
    # Create mock invoice data
    invoice_data = """invoice_id,amount,supplier,due_date,status,payment_terms,notes
INV-2025-001,5000.00,ABC Electronics,2025-07-15,paid,NET30,Regular order
INV-2025-002,8200.00,XYZ Supplies,2025-07-20,pending,NET15,Standard delivery
INV-2025-003,3500.00,TechCorp,2025-07-25,overdue,NET30,Late payment
INV-2025-004,15750.00,ABC Electronics,2025-07-30,flagged,NET30,Amount deviation - requires approval"""
    
    invoice_file = "./data/invoices/comprehensive_invoices.csv"
    os.makedirs(os.path.dirname(invoice_file), exist_ok=True)
    
    if not os.path.exists(invoice_file):
        with open(invoice_file, 'w') as f:
            f.write(invoice_data)
        print(f"✅ Created mock file: {invoice_file}")
    
    # Create mock shipment data
    shipment_data = """shipment_id,origin,destination,carrier,status,shipped_date,expected_delivery,value,weight,notes
SHP-2025-001,New York USA,London UK,Global Shipping Inc,delivered,2025-06-15,2025-06-25,12000.00,150kg,On time delivery
SHP-2025-002,Los Angeles USA,Tokyo Japan,Express Worldwide,in_transit,2025-06-20,2025-06-30,8500.00,75kg,Standard shipping
SHP-2025-003,Chicago USA,Berlin Germany,Alternative Carriers,delayed,2025-06-18,2025-06-28,6200.00,45kg,Unusual carrier selection"""
    
    shipment_file = "./data/shipments/comprehensive_shipments.csv"
    os.makedirs(os.path.dirname(shipment_file), exist_ok=True)
    
    if not os.path.exists(shipment_file):
        with open(shipment_file, 'w') as f:
            f.write(shipment_data)
        print(f"✅ Created mock file: {shipment_file}")

def main():
    """Run all tests"""
    print("🚀 Logistics Pulse Copilot - Component Test\n")
    
    # Run tests
    test_dependencies()
    test_data_directories()
    create_mock_files()
    test_component_initialization()
    
    print("\n📋 Test Summary:")
    print("- If all components show ✅, the system should work properly")
    print("- If some components show ❌, check the error messages above")
    print("- Missing dependencies can be installed with pip")
    print("- The system will run in mock mode for missing components")
    
    print("\n🚀 Ready to start the server with: python main_enhanced.py")

if __name__ == "__main__":
    main()
