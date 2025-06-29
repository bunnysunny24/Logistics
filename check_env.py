#!/usr/bin/env python3
"""
Environment Configuration Checker for Logistics Pulse Copilot
"""
import os
from pathlib import Path

def check_backend_env():
    """Check backend environment configuration"""
    backend_env = Path("backend/.env")
    backend_env_example = Path("backend/.env.example")
    
    print("🔧 Backend Environment Check:")
    
    if not backend_env.exists():
        print("  ❌ backend/.env not found")
        if backend_env_example.exists():
            print("  💡 Copy backend/.env.example to backend/.env and configure it")
        return False
    
    # Load backend env
    try:
        with open(backend_env, 'r') as f:
            content = f.read()
        
        # Check for required variables
        required_vars = [
            'OPENAI_API_KEY',
            'DATA_DIR',
            'LLM_MODEL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if f"{var}=" not in content:
                missing_vars.append(var)
        
        if missing_vars:
            print(f"  ❌ Missing variables: {', '.join(missing_vars)}")
            return False
        
        # Check if API key is set
        if 'OPENAI_API_KEY=your_openai_api_key_here' in content or 'OPENAI_API_KEY=' in content.replace('OPENAI_API_KEY=sk-', ''):
            if 'OPENAI_API_KEY=sk-' not in content:
                print("  ⚠️  OpenAI API key not configured (using placeholder)")
        else:
            print("  ✅ OpenAI API key configured")
        
        print("  ✅ backend/.env exists and is configured")
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading backend/.env: {e}")
        return False

def check_frontend_env():
    """Check frontend environment configuration"""
    frontend_env = Path("frontend/.env")
    frontend_env_example = Path("frontend/.env.example")
    
    print("\n🌐 Frontend Environment Check:")
    
    if not frontend_env.exists():
        print("  ⚠️  frontend/.env not found (will use defaults)")
        if frontend_env_example.exists():
            print("  💡 Copy frontend/.env.example to frontend/.env for custom configuration")
        return True  # Frontend .env is optional
    
    try:
        with open(frontend_env, 'r') as f:
            content = f.read()
        
        # Check API URL
        if 'REACT_APP_API_URL=' in content:
            print("  ✅ API URL configured")
        else:
            print("  ⚠️  API URL not configured (will use default)")
        
        print("  ✅ frontend/.env exists")
        return True
        
    except Exception as e:
        print(f"  ❌ Error reading frontend/.env: {e}")
        return False

def check_data_directories():
    """Check if required data directories exist"""
    print("\n📁 Data Directories Check:")
    
    required_dirs = [
        'data',
        'data/uploads',
        'data/index',
        'data/invoices',
        'data/shipments',
        'data/policies'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ⚠️  {dir_path} (will be created automatically)")
    
    return True

def main():
    print("🚀 Logistics Pulse Copilot - Environment Configuration Check")
    print("=" * 60)
    
    backend_ok = check_backend_env()
    frontend_ok = check_frontend_env()
    data_ok = check_data_directories()
    
    print("\n" + "=" * 60)
    print("📋 Summary:")
    
    if backend_ok and frontend_ok and data_ok:
        print("✅ All checks passed! Your environment is properly configured.")
        print("\n🚀 Ready to start:")
        print("   - Run: .\\start.ps1 (PowerShell) or start.bat (CMD)")
        print("   - Or manually start backend and frontend servers")
    else:
        print("⚠️  Some configuration issues found. Please review above.")
        print("\n🔧 Next steps:")
        if not backend_ok:
            print("   - Configure backend/.env with your API keys")
        print("   - Run this script again to verify configuration")

if __name__ == "__main__":
    main()
