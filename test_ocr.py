#!/usr/bin/env python3
"""
Test OCR functionality for the Royalton AI Screener
"""

def test_ocr_availability():
    """Test if OCR dependencies are working."""
    print("🧪 Testing OCR functionality...")
    
    try:
        import pytesseract
        import pdf2image
        from PIL import Image
        print("✅ OCR libraries imported successfully")
        
        # Test basic OCR functionality
        print("🔍 Testing basic OCR...")
        
        # Create a simple test image with text
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a test image
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        draw.text((10, 30), "John Smith - Front Desk Agent", fill='black')
        draw.text((10, 50), "Email: john.smith@email.com", fill='black')
        draw.text((10, 70), "Phone: (555) 123-4567", fill='black')
        
        # Save test image
        test_image_path = "test_resume_image.png"
        img.save(test_image_path)
        print(f"📄 Created test image: {test_image_path}")
        
        # Test OCR on the image
        extracted_text = pytesseract.image_to_string(img, lang='eng')
        print(f"🔍 OCR extracted text: {repr(extracted_text)}")
        
        if "john" in extracted_text.lower() and "smith" in extracted_text.lower():
            print("✅ OCR test PASSED - Text extracted successfully!")
            return True
        else:
            print("❌ OCR test FAILED - Could not extract expected text")
            return False
            
    except ImportError as e:
        print(f"❌ OCR libraries not available: {e}")
        print("📦 Install with: pip install pytesseract pdf2image pillow")
        return False
    except Exception as e:
        print(f"❌ OCR test failed: {e}")
        return False

def test_enhanced_screener_ocr():
    """Test the enhanced screener's OCR functionality."""
    try:
        from royalton_ai_screener import RoyaltonAIScreener
        
        print("🧪 Testing enhanced screener OCR integration...")
        screener = RoyaltonAIScreener()
        
        # Test the OCR extraction method directly
        if hasattr(screener, '_extract_text_with_ocr'):
            print("✅ OCR method found in screener")
            
            # Test with the created image
            if Path("test_resume_image.png").exists():
                text = screener._extract_text_with_ocr("test_resume_image.png")
                if text and len(text.strip()) > 0:
                    print(f"✅ Screener OCR working: {len(text)} characters extracted")
                    print(f"📄 Sample text: {text[:100]}...")
                    return True
                else:
                    print("❌ Screener OCR failed to extract text")
                    return False
            else:
                print("⚠ Test image not found")
                return False
        else:
            print("❌ OCR method not found in screener")
            return False
            
    except Exception as e:
        print(f"❌ Screener OCR test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("  ROYALTON AI SCREENER - OCR FUNCTIONALITY TEST")
    print("=" * 60)
    
    from pathlib import Path
    
    # Test 1: Basic OCR availability
    ocr_available = test_ocr_availability()
    
    # Test 2: Enhanced screener integration
    if ocr_available:
        screener_ocr_ok = test_enhanced_screener_ocr()
        
        if screener_ocr_ok:
            print("\n🎉 RESULT: OCR fully integrated and working!")
            print("📊 Scanned resumes will be processed automatically!")
        else:
            print("\n⚠ RESULT: OCR available but integration needs work")
    else:
        print("\n❌ RESULT: OCR not available - install dependencies first")
    
    # Cleanup
    test_file = Path("test_resume_image.png")
    if test_file.exists():
        test_file.unlink()
        print("🧹 Cleaned up test files")
    
    print("\n" + "=" * 60)
    input("Press Enter to exit...")
