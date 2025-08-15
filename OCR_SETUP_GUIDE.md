# OCR Integration Setup Guide

## Overview
The HR ATS system now includes OCR (Optical Character Recognition) capabilities to automatically process scanned resumes and image-based documents.

## Installation Steps

### 1. Python Libraries (Already Installed)
The following OCR libraries have been installed:
- `pytesseract` - Python wrapper for Tesseract OCR
- `pdf2image` - Converts PDF pages to images
- `Pillow (PIL)` - Image processing library

### 2. Tesseract OCR Engine (Required)

**Windows Installation:**

**Option A: Automated Installation**
```bash
# Run the provided installation script
Install_Tesseract_OCR.bat
```

**Option B: Manual Installation**
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Choose the Windows installer (64-bit recommended)
3. Run installer as Administrator
4. ✅ **IMPORTANT:** Check "Add to PATH" during installation
5. Default installation path: `C:\Program Files\Tesseract-OCR`

**Option C: Chocolatey (if available)**
```bash
choco install tesseract -y
```

### 3. Verify Installation
```bash
# Test Tesseract is working
tesseract --version

# Test full OCR functionality
python test_ocr_new.py
```

## How OCR Works in the System

### Automatic Detection
The system automatically detects when OCR is needed based on:
- **Low text extraction ratio** (< 10 characters per KB)
- **File size vs extracted text** mismatch
- **Failed standard text extraction**

### OCR Processing Flow
1. **PDF Documents:** Converted to high-resolution images (300 DPI)
2. **Image Enhancement:** Automatic RGB conversion and optimization
3. **Text Extraction:** Advanced OCR with English language model
4. **Quality Validation:** Text length and content verification

### Supported Formats
- ✅ **PDF files** (scanned or image-based)
- ✅ **Image files** (PNG, JPG, TIFF, BMP)
- ✅ **Multi-page PDFs** (processes first 3 pages for performance)
- ❌ **DOC/DOCX** (requires conversion to PDF first)

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"
1. **Restart your command prompt** after installation
2. **Check PATH:** Ensure `C:\Program Files\Tesseract-OCR` is in your system PATH
3. **Manual PATH setup:**
   - Open System Properties → Environment Variables
   - Add `C:\Program Files\Tesseract-OCR` to PATH
   - Restart command prompt

### OCR Not Detecting Text
1. **Check image quality:** Ensure scanned documents are clear and high-resolution
2. **File format:** Try converting to PDF or high-quality PNG
3. **Image enhancement:** The system automatically enhances images, but very poor quality scans may still fail

### Performance Considerations
- **Large PDFs:** Only first 3 pages are processed by default
- **High DPI:** 300 DPI used for accuracy (slower but better results)
- **Fallback:** If OCR fails, standard text extraction is still attempted

## Usage Examples

### Processing Scanned Resumes
```python
# The system automatically detects and processes scanned documents
screener = RoyaltonAIScreener()
results = screener.screen_candidates("input_resumes/")

# OCR will automatically trigger for:
# - Scanned PDF resumes
# - Image-based documents
# - PDFs with minimal extractable text
```

### Manual OCR Testing
```python
# Test OCR on a specific file
from royalton_ai_screener import RoyaltonAIScreener
screener = RoyaltonAIScreener()

text = screener.extract_text_simple("scanned_resume.pdf")
# OCR will be used automatically if needed
```

## OCR Output in Results

When OCR is used, you'll see additional information in the console:
```
🔍 Low text extraction detected - trying OCR fallback...
🔍 Converting PDF to images for OCR...
🔍 OCR processing page 1...
🔍 OCR processing page 2...
✅ OCR completed: 1,250 characters from 2 pages
```

The final results will include all extracted text in the Excel export with normal confidence scoring.

## Files Added for OCR

- `Install_Tesseract_OCR.bat` - Automated Tesseract installation
- `test_ocr_new.py` - Comprehensive OCR testing
- Enhanced `royalton_ai_screener.py` with OCR integration
- This documentation file

## Performance Notes

- **First-time setup:** May take a few minutes to install Tesseract
- **Processing speed:** OCR adds 5-15 seconds per scanned document
- **Accuracy:** Modern OCR is 95%+ accurate on good quality scans
- **Memory usage:** Temporarily higher during image processing

The system maintains all existing functionality while adding intelligent OCR fallback for maximum resume processing coverage.
