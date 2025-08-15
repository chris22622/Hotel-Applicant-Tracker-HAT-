#!/usr/bin/env python3
"""
Test OCR functionality for the HR ATS system.
This script tests text extraction capabilities including OCR for scanned documents.
"""

import os
import sys
from pathlib import Path

def test_ocr_dependencies():
    """Test if all OCR dependencies are available."""
    print("🔍 Testing OCR Dependencies...")
    print("=" * 50)
    
    # Test pytesseract
    try:
        import pytesseract
        print("✅ pytesseract library: Available")
        
        # Test Tesseract executable
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR engine: v{version}")
            tesseract_available = True
        except Exception as e:
            print("❌ Tesseract OCR engine: Not found in PATH")
            print(f"   Error: {e}")
            tesseract_available = False
            
    except ImportError:
        print("❌ pytesseract library: Not installed")
        tesseract_available = False
    
    # Test pdf2image
    try:
        import pdf2image
        print("✅ pdf2image library: Available")
    except ImportError:
        print("❌ pdf2image library: Not installed")
    
    # Test PIL/Pillow
    try:
        from PIL import Image
        print("✅ PIL/Pillow library: Available")
    except ImportError:
        print("❌ PIL/Pillow library: Not installed")
    
    return tesseract_available

def test_basic_functionality():
    """Test basic OCR functionality with a simple text image."""
    print("\n🧪 Testing Basic OCR Functionality...")
    print("=" * 50)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import pytesseract
        
        # Create a simple test image with text
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Use default font
        try:
            # Try to use a better font if available
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "This is a test for OCR", fill='black', font=font)
        
        # Save test image
        test_image_path = "test_ocr_image.png"
        img.save(test_image_path)
        print(f"📸 Created test image: {test_image_path}")
        
        # Test OCR
        text = pytesseract.image_to_string(img)
        print(f"🔍 OCR Result: '{text.strip()}'")
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
        if "test" in text.lower() and "ocr" in text.lower():
            print("✅ Basic OCR test: PASSED")
            return True
        else:
            print("⚠ Basic OCR test: Text not recognized correctly")
            return False
            
    except Exception as e:
        print(f"❌ Basic OCR test failed: {e}")
        return False

def test_screener_integration():
    """Test OCR integration with the AI screener."""
    print("\n🔗 Testing Screener Integration...")
    print("=" * 50)
    
    try:
        from royalton_ai_screener import RoyaltonAIScreener
        screener = RoyaltonAIScreener()
        
        # Check if OCR method exists
        if hasattr(screener, '_extract_text_with_ocr'):
            print("✅ OCR method found in screener")
            
            # Create a test resume image
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (600, 200), color='white')
            draw = ImageDraw.Draw(img)
            
            draw.text((10, 20), "JOHN SMITH", fill='black')
            draw.text((10, 50), "Front Desk Agent", fill='black')
            draw.text((10, 80), "Email: john.smith@hotel.com", fill='black')
            draw.text((10, 110), "Phone: (555) 123-4567", fill='black')
            draw.text((10, 140), "Experience: 3 years hotel front desk", fill='black')
            
            test_image = "test_resume_ocr.png"
            img.save(test_image)
            
            # Test OCR extraction
            extracted_text = screener._extract_text_with_ocr(test_image)
            
            # Clean up
            if os.path.exists(test_image):
                os.remove(test_image)
            
            if extracted_text and len(extracted_text.strip()) > 0:
                print(f"✅ Screener OCR working: {len(extracted_text)} characters")
                print(f"📄 Sample: {extracted_text[:100]}...")
                return True
            else:
                print("❌ Screener OCR failed to extract text")
                return False
        else:
            print("❌ OCR method not found in screener")
            return False
            
    except Exception as e:
        print(f"❌ Screener integration test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 HR ATS OCR Functionality Test")
    print("=" * 50)
    
    # Test dependencies
    tesseract_ok = test_ocr_dependencies()
    
    if tesseract_ok:
        # Test basic functionality
        ocr_ok = test_basic_functionality()
        
        if ocr_ok:
            # Test screener integration
            integration_ok = test_screener_integration()
            
            if integration_ok:
                print("\n🎉 OCR Setup Complete!")
                print("✅ All tests passed - OCR is ready for scanned resumes")
            else:
                print("\n⚠ Basic OCR works but integration needs fixing")
        else:
            print("\n⚠ OCR partially working but may have issues")
    else:
        print("\n❌ OCR Setup Incomplete")
        print("\n📋 To fix this:")
        print("1. Run: Install_Tesseract_OCR.bat")
        print("2. Or manually install Tesseract from:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("3. Make sure Tesseract is added to your PATH")
        print("4. Restart your command prompt and try again")
    
    print("\n" + "=" * 50)
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()
