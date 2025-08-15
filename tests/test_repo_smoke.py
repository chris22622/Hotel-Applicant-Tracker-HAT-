#!/usr/bin/env python3
"""
Smoke tests for Hotel Applicant Tracker repository
Validates basic functionality and file structure
"""

import os
import sys
import yaml
import json
from pathlib import Path


def test_repository_structure():
    """Test that essential files exist"""
    required_files = [
        'README.md',
        'requirements.txt',
        'hotel_ai_screener.py',
        'streamlit_app.py',
        'hotel_config.yaml',
        'LICENSE',
        '.gitignore',
        'Dockerfile'
    ]
    
    required_dirs = [
        'input_resumes',
        'tests',
        'assets',
        '.github/workflows'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    assert not missing_files, f"Missing required files: {missing_files}"
    assert not missing_dirs, f"Missing required directories: {missing_dirs}"
    print("✅ Repository structure is valid")


def test_config_files():
    """Test that configuration files are valid"""
    # Test hotel_config.yaml
    with open('hotel_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'positions' in config, "hotel_config.yaml missing 'positions' key"
    assert len(config['positions']) > 0, "No positions defined in config"
    
    # Test at least one position has required fields
    first_position = next(iter(config['positions'].values()))
    required_fields = ['must_have_skills', 'nice_to_have_skills', 'cultural_fit_keywords']
    for field in required_fields:
        assert field in first_position, f"Position missing required field: {field}"
    
    print("✅ Configuration files are valid")


def test_python_imports():
    """Test that main modules can be imported"""
    try:
        import hotel_ai_screener
        screener = hotel_ai_screener.HotelAIScreener()
        assert screener is not None
        print("✅ hotel_ai_screener imports successfully")
    except ImportError as e:
        print(f"❌ Failed to import hotel_ai_screener: {e}")
        raise


def test_requirements():
    """Test that requirements.txt is valid"""
    with open('requirements.txt', 'r') as f:
        requirements = f.read().strip().split('\n')
    
    # Check for essential packages
    essential_packages = [
        'streamlit',
        'pandas',
        'pyyaml',
        'spacy',
        'pytest'
    ]
    
    req_text = ' '.join(requirements).lower()
    missing_packages = []
    for package in essential_packages:
        if package not in req_text:
            missing_packages.append(package)
    
    assert not missing_packages, f"Missing essential packages: {missing_packages}"
    print("✅ Requirements file is valid")


def test_sample_data():
    """Test that sample data exists and is accessible"""
    input_dir = Path('input_resumes')
    if not input_dir.exists():
        print("⚠️  No input_resumes directory found")
        return
    
    sample_files = list(input_dir.glob('*.{txt,pdf,docx}'))
    sample_files.extend(list(input_dir.glob('*.{TXT,PDF,DOCX}')))
    
    if len(sample_files) == 0:
        print("⚠️  No sample resume files found")
        return
    
    print(f"✅ Found {len(sample_files)} sample resume files")


def test_ci_workflow():
    """Test that CI workflow file is valid"""
    ci_file = Path('.github/workflows/ci.yml')
    if not ci_file.exists():
        print("⚠️  No CI workflow file found")
        return
    
    with open(ci_file, 'r') as f:
        content = f.read()
    
    # Basic validation
    assert 'python' in content.lower(), "CI workflow should mention Python"
    assert 'pytest' in content.lower(), "CI workflow should run pytest"
    print("✅ CI workflow file exists and looks valid")


def run_all_tests():
    """Run all smoke tests"""
    tests = [
        test_repository_structure,
        test_config_files,
        test_python_imports,
        test_requirements,
        test_sample_data,
        test_ci_workflow
    ]
    
    passed = 0
    failed = 0
    
    print("🏨 Hotel Applicant Tracker - Smoke Tests")
    print("=" * 45)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 45)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All smoke tests passed! Repository is ready.")
        return 0
    else:
        print("💥 Some tests failed. Check the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
