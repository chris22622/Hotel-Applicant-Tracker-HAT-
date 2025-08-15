#!/usr/bin/env python3
"""
Check if all dependencies for Excel export are available.
Run this to diagnose Excel file creation issues.
"""

def check_dependencies():
    """Check if required packages are installed."""
    
    required_packages = {
        'pandas': 'pandas',
        'openpyxl': 'openpyxl', 
        'xlsxwriter': 'xlsxwriter'
    }
    
    missing = []
    available = []
    
    print("🔍 Checking dependencies for Excel export...\n")
    
    for package_name, import_name in required_packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name} - Available (version: {version})")
            available.append(package_name)
        except ImportError:
            print(f"❌ {package_name} - MISSING")
            missing.append(package_name)
    
    print(f"\n📊 SUMMARY:")
    print(f"   Available: {len(available)}/{len(required_packages)}")
    print(f"   Missing: {len(missing)}")
    
    if missing:
        print(f"\n⚠ Missing packages for Excel export: {', '.join(missing)}")
        print("📦 Install with:")
        print(f"   pip install {' '.join(missing)}")
        print("\n🔧 Or install all enhanced features:")
        print("   pip install -r enhanced_requirements.txt")
        return False
    else:
        print("\n🎉 All dependencies available for Excel export!")
        print("✅ Excel files (.xlsx) should be created successfully!")
        return True

def test_excel_creation():
    """Test if Excel file can actually be created."""
    try:
        import pandas as pd
        import tempfile
        from pathlib import Path
        
        print("\n🧪 Testing Excel file creation...")
        
        # Create test data
        test_data = pd.DataFrame({
            'Name': ['Test Candidate'],
            'Phone': ['555-1234'],
            'Email': ['test@email.com'],
            'Score': [85.5]
        })
        
        # Try to create Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
            test_data.to_excel(writer, sheet_name='Test', index=False)
        
        # Check if file was created
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            print("✅ Excel file creation test: SUCCESS")
            tmp_path.unlink()  # Clean up
            return True
        else:
            print("❌ Excel file creation test: FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Excel file creation test: FAILED - {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  ROYALTON AI SCREENER - DEPENDENCY CHECKER")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    
    if deps_ok:
        excel_ok = test_excel_creation()
        
        if excel_ok:
            print("\n🎉 RESULT: Excel export should work perfectly!")
            print("📊 Run the screener - you'll get proper .xlsx files!")
        else:
            print("\n⚠ RESULT: Dependencies installed but Excel creation failed!")
            print("🔧 Try: pip install --upgrade pandas openpyxl")
    else:
        print("\n❌ RESULT: Missing dependencies - install them first!")
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")
