#!/usr/bin/env python3
"""
Quick start script for the RAG system.
Automatically sets up and runs the system.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def setup_virtual_environment():
    """Set up virtual environment if it doesn't exist."""
    venv_path = Path("rag_env")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("🔄 Creating virtual environment...")
    if run_command("python -m venv rag_env", "Creating virtual environment"):
        print("✅ Virtual environment created")
        return True
    return False

def install_dependencies():
    """Install required dependencies."""
    # Determine the correct pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "rag_env\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        pip_cmd = "rag_env/bin/pip"
    
    if run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        print("✅ Dependencies installed")
        return True
    return False

def setup_environment():
    """Set up environment configuration."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ Environment file already exists")
        return True
    
    env_example = Path("env.example")
    if env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        print("✅ Environment file created from template")
        print("⚠️  Please edit .env file with your API keys before running the system")
        return True
    else:
        print("❌ No env.example template found")
        return False

def initialize_system():
    """Initialize the RAG system."""
    if run_command("python scripts/init_db.py", "Initializing system"):
        print("✅ System initialized")
        return True
    return False

def run_demo():
    """Run the demo to test the system."""
    print("🔄 Running demo...")
    if run_command("python scripts/demo.py", "Running demo"):
        print("✅ Demo completed successfully")
        return True
    return False

def start_system():
    """Start the RAG system."""
    print("🚀 Starting RAG system...")
    print("📖 API docs will be available at: http://localhost:8000/docs")
    print("🌐 Web interface will be available at: http://localhost:8000")
    print("⏹️  Press Ctrl+C to stop the system")
    
    try:
        # Determine the correct python command based on OS
        if os.name == 'nt':  # Windows
            python_cmd = "rag_env\\Scripts\\python"
        else:  # Unix/Linux/Mac
            python_cmd = "rag_env/bin/python"
        
        subprocess.run(f"{python_cmd} main.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n🛑 System stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start system: {e}")

def main():
    """Main quick start function."""
    print("🚀 RAG System Quick Start")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Setup steps
    steps = [
        ("Setting up virtual environment", setup_virtual_environment),
        ("Installing dependencies", install_dependencies),
        ("Setting up environment", setup_environment),
        ("Initializing system", initialize_system),
    ]
    
    # Execute setup steps
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            print("Please check the error messages above and try again")
            return
    
    # Ask if user wants to run demo
    print("\n" + "=" * 50)
    demo_choice = input("Would you like to run a demo to test the system? (y/n): ").lower().strip()
    
    if demo_choice in ['y', 'yes']:
        if not run_demo():
            print("❌ Demo failed, but you can still try starting the system")
    
    # Ask if user wants to start the system
    print("\n" + "=" * 50)
    start_choice = input("Would you like to start the RAG system now? (y/n): ").lower().strip()
    
    if start_choice in ['y', 'yes']:
        start_system()
    else:
        print("\n✅ Setup completed successfully!")
        print("\nTo start the system later:")
        print("1. Activate virtual environment:")
        if os.name == 'nt':  # Windows
            print("   rag_env\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            print("   source rag_env/bin/activate")
        print("2. Run: python main.py")
        print("3. Open http://localhost:8000 in your browser")

if __name__ == "__main__":
    main()
