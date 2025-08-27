#!/usr/bin/env python3
"""
Database initialization script for the RAG system.
"""
import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings

def create_directories():
    """Create necessary directories for the RAG system."""
    directories = [
        settings.upload_dir,
        settings.chunk_dir,
        os.path.dirname(settings.faiss_index_path),
        "./data/export",
        "./logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_template = Path("env.example")
    
    if not env_file.exists() and env_template.exists():
        import shutil
        shutil.copy(env_template, env_file)
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys before running the system")
    elif env_file.exists():
        print("✓ .env file already exists")
    else:
        print("⚠️  No env.example template found")

def check_dependencies():
    """Check if required dependencies are available."""
    print("\nChecking dependencies...")
    
    try:
        import fastapi
        print("✓ FastAPI")
    except ImportError:
        print("✗ FastAPI - run: pip install fastapi")
    
    try:
        import sentence_transformers
        print("✓ Sentence Transformers")
    except ImportError:
        print("✗ Sentence Transformers - run: pip install sentence-transformers")
    
    try:
        import faiss
        print("✓ FAISS")
    except ImportError:
        print("✗ FAISS - run: pip install faiss-cpu")
    
    try:
        import openai
        print("✓ OpenAI")
    except ImportError:
        print("✗ OpenAI - run: pip install openai")
    
    try:
        import PyPDF2
        print("✓ PyPDF2")
    except ImportError:
        print("✗ PyPDF2 - run: pip install PyPDF2")
    
    try:
        import beautifulsoup4
        print("✓ BeautifulSoup4")
    except ImportError:
        print("✗ BeautifulSoup4 - run: pip install beautifulsoup4")

def main():
    """Main initialization function."""
    print("🚀 RAG System Initialization")
    print("=" * 40)
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Create environment file
    print("\nSetting up environment...")
    create_env_file()
    
    # Check dependencies
    check_dependencies()
    
    print("\n" + "=" * 40)
    print("✅ Initialization complete!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Install missing dependencies: pip install -r requirements.txt")
    print("3. Run the system: python -m app.main")
    print("4. Open http://localhost:8000 in your browser")

if __name__ == "__main__":
    main()
