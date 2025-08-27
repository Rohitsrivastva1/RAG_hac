#!/usr/bin/env python3
"""
Simple server runner that keeps the server alive.
"""
import subprocess
import sys
import time
import signal
import os

def signal_handler(sig, frame):
    print('\n🛑 Shutting down server...')
    sys.exit(0)

def main():
    print("🚀 Starting RAG System Server...")
    print("📖 Open http://localhost:8000/docs for API documentation")
    print("🌐 Open http://localhost:8000 for web interface")
    print("🔄 Server will keep running until you stop it (Ctrl+C)")
    print("")
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start the server
        process = subprocess.Popen([sys.executable, "main.py"])
        
        print("✅ Server started successfully!")
        print("🌐 Web Interface: http://localhost:8000")
        print("📖 API Docs: http://localhost:8000/docs")
        print("")
        print("Press Ctrl+C to stop the server")
        
        # Keep the script running
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("❌ Server process terminated unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping server...")
        if 'process' in locals():
            process.terminate()
            process.wait()
        print("✅ Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
