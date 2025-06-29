#!/usr/bin/env python3
"""
Simple startup script for the Logistics Pulse Copilot API
This avoids reloader issues and provides better error handling
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def main():
    """Start the server with proper error handling"""
    print("🚀 Starting Logistics Pulse Copilot API (Production Mode)")
    
    # Set environment variables for stable operation
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow warnings
    
    try:
        # Import the app after setting environment
        from main_enhanced import app
        
        # Start server without reloader for stability
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reloader to avoid multiprocessing issues
            log_level="info",
            access_log=True
        )
        
    except ImportError as e:
        print(f"❌ Failed to import main application: {e}")
        print("🔧 Try running: python test_components.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
