#!/usr/bin/env python3
"""
Enhanced setup and validation script for Logistics Pulse Copilot
Validates the improved RAG implementation, anomaly detection, and data alignment
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_step(step, description):
    """Print a formatted step"""
    print(f"\n🔧 Step {step}: {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} is supported")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} is not supported. Requires Python 3.8+")
        return False

def check_environment_variables():
    """Check and set up environment variables"""
    print("🔑 Checking environment variables...")
    
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for enhanced LLM functionality",
        "DATA_DIR": "Data directory path (optional, defaults to ./data)",
        "LLM_MODEL": "LLM model name (optional, defaults to gpt-4)",
        "EMBEDDING_MODEL": "Embedding model name (optional, defaults to text-embedding-3-small)"
    }
    
    env_status = {}
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if var == "OPENAI_API_KEY":
                print(f"  ✅ {var}: Set (hidden for security)")
            else:
                print(f"  ✅ {var}: {value}")
            env_status[var] = True
        else:
            if var == "OPENAI_API_KEY":
                print(f"  ⚠️ {var}: Not set - {description}")
                print(f"      System will run in demo mode without full AI capabilities")
            else:
                print(f"  ℹ️ {var}: Not set - {description}")
            env_status[var] = False
    
    return env_status

def setup_directory_structure():
    """Set up the required directory structure"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "./data",
        "./data/invoices",
        "./data/shipments", 
        "./data/policies",
        "./data/anomalies",
        "./data/index",
        "./data/uploads",
        "./backend/models",
        "./backend/pipeline",
        "./backend/utils",
        "./backend/prompts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created/verified: {directory}")
    
    return True

def validate_data_files():
    """Validate that required data files exist and are properly formatted"""
    print("📊 Validating data files...")
    
    required_files = {
        "./data/invoices/comprehensive_invoices.csv": "Main invoice dataset",
        "./data/shipments/comprehensive_shipments.csv": "Main shipment dataset",
        "./data/policies/payout-rules-v3.md": "Payment policy document",
        "./data/policies/shipment-guidelines-v2.md": "Shipment guidelines document"
    }
    
    validation_results = {}
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    print(f"  ✅ {file_path}: {len(df)} records loaded")
                    
                    # Validate required columns
                    if 'invoice' in file_path:
                        required_cols = ['invoice_id', 'supplier', 'amount', 'issue_date', 'due_date']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"     ⚠️ Missing columns: {missing_cols}")
                        else:
                            print(f"     ✅ All required invoice columns present")
                    
                    elif 'shipment' in file_path:
                        required_cols = ['shipment_id', 'origin', 'destination', 'carrier', 'departure_date']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            print(f"     ⚠️ Missing columns: {missing_cols}")
                        else:
                            print(f"     ✅ All required shipment columns present")
                            
                elif file_path.endswith('.md'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"  ✅ {file_path}: {len(content)} characters loaded")
                    
                validation_results[file_path] = True
                
            except Exception as e:
                print(f"  ❌ {file_path}: Error reading file - {e}")
                validation_results[file_path] = False
        else:
            print(f"  ❌ {file_path}: File not found - {description}")
            validation_results[file_path] = False
    
    return validation_results

def install_dependencies():
    """Install required Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("  ❌ requirements.txt not found")
            return False
        
        # Install dependencies
        print("  🔄 Running pip install -r requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Dependencies installed successfully")
            return True
        else:
            print(f"  ❌ Error installing dependencies: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error during installation: {e}")
        return False

def test_component_imports():
    """Test that all enhanced components can be imported"""
    print("🧪 Testing component imports...")
    
    components = {
        "RAG Model": ("backend.models.rag_model", "LogisticsPulseRAG"),
        "Anomaly Detector": ("backend.pipeline.enhanced_anomaly_detector", "EnhancedAnomalyDetector"),
        "Document Processor": ("backend.utils.document_processor", "DocumentProcessor")
    }
    
    import_results = {}
    
    # Add backend to path temporarily
    sys.path.insert(0, './backend')
    
    try:
        for component_name, (module_name, class_name) in components.items():
            try:
                module = __import__(module_name, fromlist=[class_name])
                component_class = getattr(module, class_name)
                print(f"  ✅ {component_name}: Import successful")
                import_results[component_name] = True
            except ImportError as e:
                print(f"  ❌ {component_name}: Import failed - {e}")
                import_results[component_name] = False
            except Exception as e:
                print(f"  ❌ {component_name}: Error - {e}")
                import_results[component_name] = False
    finally:
        # Remove backend from path
        sys.path.remove('./backend')
    
    return import_results

def test_rag_improvements():
    """Test the enhanced RAG model functionality"""
    print("🤖 Testing RAG model improvements...")
    
    sys.path.insert(0, './backend')
    
    try:
        from models.rag_model import LogisticsPulseRAG
        
        # Initialize RAG model
        rag = LogisticsPulseRAG()
        print("  ✅ RAG model initialized")
        
        # Test status
        status = rag.get_status()
        print(f"  📊 Model: {status.get('model', 'unknown')}")
        print(f"  📊 Embedding model: {status.get('embedding_model', 'unknown')}")
        print(f"  📊 API configured: {status.get('api_key_configured', False)}")
        print(f"  📊 Vector stores: {len(status.get('vector_stores', {}))}")
        
        # Test query processing
        test_queries = [
            "What are the main invoice compliance issues?",
            "Show me high-risk shipment anomalies",
            "Which suppliers have payment term violations?"
        ]
        
        for query in test_queries:
            try:
                result = rag.process_query(query)
                confidence = result.get('confidence', 0)
                sources = len(result.get('sources', []))
                print(f"  ✅ Query processed: confidence={confidence:.2f}, sources={sources}")
            except Exception as e:
                print(f"  ⚠️ Query processing issue: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RAG testing failed: {e}")
        return False
    finally:
        if './backend' in sys.path:
            sys.path.remove('./backend')

def test_anomaly_detection_improvements():
    """Test the enhanced anomaly detection"""
    print("🔍 Testing anomaly detection improvements...")
    
    sys.path.insert(0, './backend')
    
    try:
        from pipeline.enhanced_anomaly_detector import EnhancedAnomalyDetector
        
        # Initialize detector
        detector = EnhancedAnomalyDetector(data_dir="./data")
        print("  ✅ Enhanced anomaly detector initialized")
        
        # Test configuration loading
        config = detector.config
        print(f"  📊 Invoice rules loaded: {len(config.get('invoice_rules', {}))}")
        print(f"  📊 Shipment rules loaded: {len(config.get('shipment_rules', {}))}")
        print(f"  📊 Fraud patterns loaded: {len(config.get('fraud_patterns', {}))}")
        
        # Test with enhanced invoice data
        test_invoice = {
            "invoice_id": "TEST-ENHANCED-001",
            "supplier": "ABC Electronics",
            "amount": 50000.0,  # High amount
            "currency": "USD",
            "issue_date": "2025-06-29",
            "due_date": "2025-09-29",  # Very long payment terms
            "payment_terms": "NET90",
            "early_discount": 0.0,
            "status": "pending",
            "approver": "clerk"  # Insufficient approval
        }
        
        anomalies = detector.detect_invoice_anomalies(test_invoice)
        print(f"  📊 Invoice anomalies detected: {len(anomalies)}")
        
        for anomaly in anomalies:
            print(f"     - {anomaly.anomaly_type}: Risk {anomaly.risk_score:.2f}")
            print(f"       {anomaly.description}")
        
        # Test with enhanced shipment data
        test_shipment = {
            "shipment_id": "TEST-ENHANCED-SHP-001",
            "origin": "New York USA",
            "destination": "London UK", 
            "carrier": "Suspicious Logistics",  # Should trigger blacklist check
            "departure_date": "2025-06-29",
            "estimated_arrival": "2025-07-20",  # Very long transit
            "actual_arrival": None,
            "status": "In Transit"
        }
        
        shipment_anomalies = detector.detect_shipment_anomalies(test_shipment)
        print(f"  📊 Shipment anomalies detected: {len(shipment_anomalies)}")
        
        for anomaly in shipment_anomalies:
            print(f"     - {anomaly.anomaly_type}: Risk {anomaly.risk_score:.2f}")
            print(f"       {anomaly.description}")
        
        # Test summary
        summary = detector.get_anomalies_summary()
        print(f"  📈 Total anomalies in system: {summary.get('total_anomalies', 0)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Anomaly detection testing failed: {e}")
        return False
    finally:
        if './backend' in sys.path:
            sys.path.remove('./backend')

def test_api_integration():
    """Test API integration and endpoints"""
    print("🌐 Testing API integration...")
    
    try:
        # Try to import the main API module
        sys.path.insert(0, './backend')
        
        try:
            import main_enhanced
            print("  ✅ Enhanced API module can be imported")
        except ImportError as e:
            print(f"  ⚠️ Could not import enhanced API module: {e}")
            try:
                import main_simple
                print("  ✅ Fallback to simple API module")
            except ImportError as e2:
                print(f"  ❌ Could not import any API module: {e2}")
                return False
        
        print("  ℹ️ To test live API endpoints, run: python backend/main_enhanced.py")
        print("  ℹ️ Then visit: http://localhost:8000/docs for API documentation")
        
        return True
        
    except Exception as e:
        print(f"  ❌ API integration test failed: {e}")
        return False
    finally:
        if './backend' in sys.path:
            sys.path.remove('./backend')

def generate_report(test_results):
    """Generate a comprehensive setup report"""
    print_header("Setup and Validation Report")
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name}: {status}")
            total_tests += 1
            if result:
                passed_tests += 1
        elif isinstance(result, dict):
            sub_passed = sum(1 for r in result.values() if r)
            sub_total = len(result)
            print(f"{test_name}: {sub_passed}/{sub_total} components working")
            total_tests += sub_total
            passed_tests += sub_passed
    
    print(f"\nOverall Score: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 Perfect setup! All components are working correctly.")
        print("Your enhanced Logistics Pulse Copilot is ready for production use.")
    elif passed_tests >= total_tests * 0.8:
        print("\n🟢 Good setup! Most components are working.")
        print("Consider configuring the missing components for full functionality.")
    elif passed_tests >= total_tests * 0.6:
        print("\n🟡 Partial setup. Core functionality is available.")
        print("Some advanced features may not work without additional configuration.")
    else:
        print("\n🔴 Setup issues detected. Please review the failed components.")
        print("Check dependencies, file paths, and configuration.")
    
    # Provide specific recommendations
    print("\n📋 Recommendations:")
    
    if not test_results.get("env_vars", {}).get("OPENAI_API_KEY", False):
        print("  • Set OPENAI_API_KEY environment variable for full AI capabilities")
    
    if not all(test_results.get("data_validation", {}).values()):
        print("  • Ensure all required data files are present and properly formatted")
    
    if not all(test_results.get("component_imports", {}).values()):
        print("  • Check that all Python dependencies are correctly installed")
    
    print("  • Run 'python test_enhanced_system.py' for detailed component testing")
    print("  • Start the API server with 'python backend/main_enhanced.py'")
    print("  • Visit http://localhost:8000/docs for interactive API documentation")

def main():
    """Main setup and validation function"""
    print_header("Logistics Pulse Copilot Enhanced Setup & Validation")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Platform: {platform.system()} {platform.release()}")
    print(f"📍 Working directory: {os.getcwd()}")
    
    test_results = {}
    
    # Step 1: Check Python version
    print_step(1, "Python Version Check")
    test_results["python_version"] = check_python_version()
    
    # Step 2: Check environment variables
    print_step(2, "Environment Variables Check")
    test_results["env_vars"] = check_environment_variables()
    
    # Step 3: Setup directory structure
    print_step(3, "Directory Structure Setup")
    test_results["directories"] = setup_directory_structure()
    
    # Step 4: Validate data files
    print_step(4, "Data Files Validation")
    test_results["data_validation"] = validate_data_files()
    
    # Step 5: Install dependencies
    print_step(5, "Dependencies Installation")
    test_results["dependencies"] = install_dependencies()
    
    # Step 6: Test component imports
    print_step(6, "Component Import Testing")
    test_results["component_imports"] = test_component_imports()
    
    # Step 7: Test RAG improvements
    print_step(7, "RAG Model Testing")
    test_results["rag_testing"] = test_rag_improvements()
    
    # Step 8: Test anomaly detection improvements
    print_step(8, "Anomaly Detection Testing")
    test_results["anomaly_testing"] = test_anomaly_detection_improvements()
    
    # Step 9: Test API integration
    print_step(9, "API Integration Testing")
    test_results["api_integration"] = test_api_integration()
    
    # Generate final report
    generate_report(test_results)
    
    return test_results

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    total_success = all(
        result if isinstance(result, bool) else all(result.values()) if isinstance(result, dict) else False
        for result in results.values()
    )
    
    sys.exit(0 if total_success else 1)
