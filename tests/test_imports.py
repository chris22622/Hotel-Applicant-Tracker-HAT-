#!/usr/bin/env python3
"""Basic tests for hotel_ai_screener module."""

import pytest
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_hotel_ai_screener():
    """Test that the main module can be imported."""
    try:
        import hotel_ai_screener
        assert hasattr(hotel_ai_screener, 'HotelAIScreener')
        print("✅ hotel_ai_screener module imported successfully")
        return True
    except ImportError as e:
        pytest.fail(f"Failed to import hotel_ai_screener: {e}")
        return False


def test_hotel_ai_screener_class():
    """Test that HotelAIScreener class can be instantiated."""
    try:
        from hotel_ai_screener import HotelAIScreener
        screener = HotelAIScreener()
        assert screener is not None
        assert hasattr(screener, 'get_hotel_job_intelligence')
        assert hasattr(screener, 'run_ai_screening')
        print("✅ HotelAIScreener class instantiated successfully")
        return True
    except Exception as e:
        pytest.fail(f"Failed to instantiate HotelAIScreener: {e}")
        return False


def test_job_intelligence():
    """Test that job intelligence data is available."""
    try:
        from hotel_ai_screener import HotelAIScreener
        screener = HotelAIScreener()
        jobs = screener.get_hotel_job_intelligence()
        
        assert isinstance(jobs, dict)
        assert len(jobs) > 0
        assert 'Front Desk Agent' in jobs
        assert 'Chef' in jobs or 'Executive Chef' in jobs
        
        # Test structure of a job definition
        job_def = list(jobs.values())[0]
        required_keys = ['must_have_skills', 'nice_to_have_skills', 'cultural_fit_keywords']
        for key in required_keys:
            assert key in job_def, f"Missing required key: {key}"
        
        print(f"✅ Job intelligence loaded with {len(jobs)} positions")
        return True
    except Exception as e:
        pytest.fail(f"Failed to load job intelligence: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    try:
        from hotel_ai_screener import HotelAIScreener
        screener = HotelAIScreener()
        config = screener.config
        
        assert isinstance(config, dict)
        assert 'positions' in config
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        pytest.fail(f"Failed to load configuration: {e}")
        return False


if __name__ == "__main__":
    """Run basic tests when executed directly."""
    print("🧪 Running Hotel AI Screener Tests")
    print("=" * 50)
    
    tests = [
        test_import_hotel_ai_screener,
        test_hotel_ai_screener_class,
        test_job_intelligence,
        test_config_loading,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
