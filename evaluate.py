#!/usr/bin/env python3
"""
Quick Evaluation Script for Hotel-Applicant-Tracker-HAT-
Run this to see the system in action with sample data in 30 seconds.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import streamlit  # noqa: F401
        import pandas  # noqa: F401
        import spacy  # noqa: F401
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False

def check_sample_resumes():
    """Check if sample resumes exist"""
    resume_dir = Path("input_resumes")
    if not resume_dir.exists():
        print("❌ No input_resumes folder found")
        return False
    
    resumes = list(resume_dir.glob("*.{txt,pdf,docx}"))
    if len(resumes) == 0:
        print("❌ No sample resumes found in input_resumes/")
        return False
    
    print(f"✅ Found {len(resumes)} sample resumes")
    return True

def run_quick_demo():
    """Run a quick command-line demo"""
    print("\n🚀 Running quick demo...")
    try:
        # Import and run a quick screening
        from hotel_ai_screener import HotelAIScreener
        
        screener = HotelAIScreener()
        print("✅ Hotel AI Screener initialized")
        
        # Check if intelligence system works
        positions = screener.get_hotel_job_intelligence()
        if positions and len(positions) > 0:
            print("✅ Hotel position intelligence loaded")
        
        print("\n🎯 System ready for screening!")
        return True
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("\n🌐 Launching web interface...")
    print("📝 Instructions:")
    print("  1. Upload resumes using the file uploader")
    print("  2. Select a hotel position from the dropdown") 
    print("  3. Click 'Run Screening' to analyze candidates")
    print("  4. Download the Excel report when complete")
    print("\n🔗 Opening Streamlit app...")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                      check=False)
    except KeyboardInterrupt:
        print("\n👋 Web interface closed")

def main():
    """Main evaluation workflow"""
    print("🏨 HOTEL-APPLICANT-TRACKER-HAT- - QUICK EVALUATION")
    print("=" * 55)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check sample data
    if not check_sample_resumes():
        print("\n💡 Tip: Add some sample resumes to input_resumes/ folder")
        print("   Supported formats: PDF, DOCX, TXT")
    
    # Run quick demo
    if not run_quick_demo():
        print("\n⚠️  Command-line demo had issues, but web interface may still work")
    
    # Ask user preference
    print("\n" + "=" * 55)
    print("🎯 EVALUATION OPTIONS:")
    print("  1. Launch web interface (recommended)")
    print("  2. Run command-line screener") 
    print("  3. Exit")
    
    try:
        choice = input("\nChoice (1-3): ").strip()
        
        if choice == "1":
            launch_web_interface()
        elif choice == "2":
            print("\n💻 Run: python hotel_ai_screener.py")
        else:
            print("\n👋 Thanks for evaluating Hotel Applicant Tracker!")
            
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation complete!")

if __name__ == "__main__":
    main()
