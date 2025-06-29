#!/usr/bin/env python3
"""
Logistics Pulse Copilot Startup Script
=====================================

This script helps you start the enhanced system with causal reasoning
and risk-based holds functionality.
"""

import os
import sys
import subprocess
import time
import requests
import webbrowser
from pathlib import Path

def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║              🚀 LOGISTICS PULSE COPILOT v2.0                 ║
    ║                                                               ║
    ║              ✨ Enhanced with Causal Reasoning               ║
    ║              🔒 Fully Local - No External APIs               ║
    ║              🧠 Risk-Based Holds & Analysis                  ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python packages
    required_packages = ['fastapi', 'uvicorn', 'pandas', 'langchain', 'requests']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    # Check Node.js for frontend
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Node.js {result.stdout.strip()}")
        else:
            print("   ❌ Node.js not found")
            return False
    except FileNotFoundError:
        print("   ❌ Node.js not found")
        return False
    
    return True

def start_backend():
    """Start the backend server"""
    print("\n🚀 Starting Backend Server...")
    
    backend_dir = Path(__file__).parent / "backend"
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return None
    
    try:
        # Start backend in background
        process = subprocess.Popen(
            [sys.executable, "main_enhanced.py"],
            cwd=backend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for startup
        time.sleep(3)
        
        # Check if backend is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("   ✅ Backend started successfully on http://localhost:8000")
                return process
            else:
                print(f"   ❌ Backend health check failed: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Backend not responding: {e}")
            return None
            
    except Exception as e:
        print(f"   ❌ Failed to start backend: {e}")
        return None

def start_frontend():
    """Start the frontend development server"""
    print("\n🎨 Starting Frontend Server...")
    
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return None
    
    # Check if node_modules exists
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("   📦 Installing frontend dependencies...")
        try:
            subprocess.run(['npm', 'install'], cwd=frontend_dir, check=True)
            print("   ✅ Dependencies installed")
        except subprocess.CalledProcessError:
            print("   ❌ Failed to install dependencies")
            return None
    
    try:
        # Start frontend
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print("   ✅ Frontend starting... (this may take a moment)")
        return process
        
    except Exception as e:
        print(f"   ❌ Failed to start frontend: {e}")
        return None

def run_demo():
    """Run the causal reasoning demo"""
    print("\n🎯 Running Causal Reasoning Demo...")
    
    try:
        subprocess.run([sys.executable, "demo_causal_flow.py"], check=True)
    except subprocess.CalledProcessError:
        print("   ⚠️  Demo script encountered an issue (this is normal if backend is starting)")
    except FileNotFoundError:
        print("   ⚠️  Demo script not found")

def main():
    """Main startup sequence"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        return
    
    print("\n✅ All dependencies satisfied!")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend. Check the logs for details.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend. Backend is still running.")
        return
    
    # Wait for frontend to be ready
    print("\n⏳ Waiting for frontend to be ready...")
    time.sleep(10)
    
    # Try to open browser
    try:
        webbrowser.open("http://localhost:3000")
        print("   🌐 Opened browser to http://localhost:3000")
    except:
        print("   🌐 Please open http://localhost:3000 in your browser")
    
    print("\n" + "="*60)
    print("🎉 SYSTEM STARTUP COMPLETE!")
    print("="*60)
    print("🔗 Frontend: http://localhost:3000")
    print("🔗 Backend API: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\n📋 DEMO FEATURES TO TRY:")
    print("   • Chat Interface: Ask about anomalies")
    print("   • Anomaly Dashboard: Check the Risk-Based Holds tab")
    print("   • Query examples:")
    print("     - 'Show me detected anomalies'")
    print("     - 'Why was invoice INV-2025-004 flagged?'")
    print("     - 'Explain the causal relationships'")
    
    print("\n🛑 To stop the servers, press Ctrl+C")
    
    try:
        # Keep running until user interrupts
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down servers...")
        
        if backend_process:
            backend_process.terminate()
            print("   ✅ Backend stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("   ✅ Frontend stopped")
        
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
