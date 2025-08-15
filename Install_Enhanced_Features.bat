@echo off
echo ===============================================
echo   INSTALL ENHANCED AI SCREENER FEATURES
echo   OCR + Semantic Matching + Web Interface
echo ===============================================
echo.

echo Installing enhanced requirements...
pip install -r enhanced_requirements.txt

echo.
echo Installing spaCy English model for semantic matching...
python -m spacy download en_core_web_sm

echo.
echo Downloading Tesseract OCR (if not installed)...
echo Note: You may need to install Tesseract manually:
echo https://github.com/UB-Mannheim/tesseract/wiki

echo.
echo ===============================================
echo   INSTALLATION COMPLETE!
echo ===============================================
echo.
echo Enhanced Features Available:
echo ✅ OCR for scanned resumes
echo ✅ Semantic skill matching with spaCy
echo ✅ YAML configuration files
echo ✅ CSV export capabilities
echo ✅ Streamlit web interface
echo.
echo Run: Royalton_AI_Web.bat for web interface
echo Run: Royalton_AI_Screener.bat for command line
echo.
pause
