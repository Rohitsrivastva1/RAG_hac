#!/usr/bin/env python3
"""
Main entry point for the RAG system.
"""
import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🚀 Starting RAG System...")
    print("📖 Open http://localhost:8000/docs for API documentation")
    print("🌐 Open http://localhost:8000 for web interface")
    print("🔄 Server will keep running until you stop it (Ctrl+C)")
    print("")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload for stability
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        input("Press Enter to exit...")
