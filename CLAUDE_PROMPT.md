# HR Applicant Tracker (HAT) - Claude AI Assistant Prompt

## Project Overview
I have a complete HR Applicant Tracking System built specifically for Royalton Resort. The system is a FREE, local-first solution that processes resumes using AI to automatically select the best candidates for hospitality positions.

## Current Status
✅ **COMPLETED FEATURES:**
- Full AI-powered resume screening with smart candidate matching
- Excel export with HR contact sheets (names, phones, emails, confidence scores)
- OCR integration for scanned/image-based resumes (libraries installed)
- Multi-format support: PDF, DOCX, TXT, images
- 118+ pre-configured Royalton Resort positions
- Batch processing capabilities
- 100% local operation (no cloud dependencies)
- Clean, type-safe code with no errors

## Key Files
- `royalton_ai_screener.py` - Main AI screening application
- `enhanced_requirements.txt` - Python package dependencies
- `Install_Enhanced_Features.bat` - One-click setup script
- `Royalton_AI_Screener.bat` - Easy launcher
- `input_resumes/` - Folder for resume files
- `screening_results/` - Output folder with Excel results

## Technical Stack
- **Language:** Python 3.12
- **Key Libraries:** pandas, openpyxl, pdfminer, python-docx, pytesseract, Pillow
- **AI Features:** Role-specific matching, skill detection, experience analysis
- **OCR:** Automatic fallback for scanned documents (Tesseract engine required)
- **Output:** Excel files with candidate contact information

## Workflow
1. Put resumes in `input_resumes/` folder
2. Run `python royalton_ai_screener.py` or `Royalton_AI_Screener.bat`
3. Enter position (e.g., "Front Desk Agent") and number needed
4. Get organized results in Excel format with top candidates

## Current Challenge
OCR libraries are installed but Tesseract OCR engine needs installation for complete scanned document processing. The system works perfectly for text-based documents.

## Goals
- Maintain enterprise-grade code quality
- Ensure portability across different Windows environments
- Provide seamless HR workflow integration
- Support both technical and non-technical users

---

**Please help me with any development, deployment, or enhancement needs for this AI-powered HR system.**
