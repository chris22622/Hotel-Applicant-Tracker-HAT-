#!/usr/bin/env python3
"""
Test script for Enhanced Hotel AI Resume Screener
Quick verification that all components are working
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import pandas
        print("✅ pandas")
    except ImportError:
        print("❌ pandas - install with: pip install pandas")
        return False
    
    try:
        import yaml
        print("✅ yaml") 
    except ImportError:
        print("❌ yaml - install with: pip install pyyaml")
        return False
    
    try:
        from enhanced_ai_screener import EnhancedHotelAIScreener
        print("✅ enhanced_ai_screener")
    except ImportError as e:
        print(f"⚠️ enhanced_ai_screener - {e}")
        return False
    
    try:
        import streamlit
        print("✅ streamlit")
    except ImportError:
        print("⚠️ streamlit - install with: pip install streamlit")
    
    return True

def test_screener_initialization():
    """Test that the screener can be initialized."""
    print("\n🧪 Testing screener initialization...")
    
    try:
        from enhanced_ai_screener import EnhancedHotelAIScreener
        screener = EnhancedHotelAIScreener()
        print("✅ EnhancedHotelAIScreener initialized")
        
        # Test position intelligence
        positions = list(screener.position_intelligence.keys())
        print(f"✅ Found {len(positions)} positions available")
        
        return True
    except Exception as e:
        print(f"❌ Screener initialization failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist."""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = ['input_resumes', 'screening_results']
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"📁 Creating {dir_name}/")
            dir_path.mkdir(exist_ok=True)
            print(f"✅ {dir_name}/ created")
    
    return True

def test_sample_processing():
    """Test with a sample resume text."""
    print("\n🧪 Testing sample resume processing...")
    
    try:
        from enhanced_ai_screener import EnhancedHotelAIScreener
        screener = EnhancedHotelAIScreener()
        
        # Sample resume text
        sample_text = """
        John Smith
        john.smith@email.com
        (555) 123-4567
        New York, NY
        
        Experience:
        Front Desk Agent - Marriott Hotel (2020-2023)
        - Handled guest check-ins and check-outs
        - Managed reservations using Opera PMS
        - Provided excellent customer service
        - Resolved guest complaints
        
        Skills:
        - Customer service
        - Opera PMS
        - Communication
        - Problem solving
        - Multilingual (English, Spanish)
        """
        
        # Test candidate info extraction
        candidate_info = screener._extract_candidate_info(sample_text)
        print(f"✅ Extracted candidate info: {candidate_info['name']}")
        
        # Test skill extraction
        skill_analysis = screener.enhanced_skill_extraction(sample_text, "Front Desk Agent")
        print(f"✅ Found {len(skill_analysis['skills'])} skills")
        
        # Test experience analysis
        exp_analysis = screener.advanced_experience_analysis(sample_text, "Front Desk Agent")
        print(f"✅ Experience analysis: {exp_analysis['experience_quality']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample processing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏨 Hotel AI Resume Screener - System Test")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_screener_initialization()
    all_passed &= test_directory_structure()
    all_passed &= test_sample_processing()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Place resume files in 'input_resumes/' folder")
        print("2. Run 'Quick_Start_Screener.bat' for web interface")
        print("3. Or run 'Enhanced_CLI_Screener.bat' for command line")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("Try running 'Quick_Setup.bat' to install missing dependencies.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    input("\nPress Enter to continue...")
    sys.exit(0 if success else 1)
