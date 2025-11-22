"""
Application Runner
==================
Simple script to run the Streamlit application.

Usage:
    python run.py
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit application."""
    
    # Ensure we're in the correct directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    # Run Streamlit
    print("🚀 Starting Student Career Prediction System...")
    print("📱 Access the app at: http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app.py",
        "--logger.level=info"
    ])

if __name__ == "__main__":
    main()
