@echo off
echo ===============================================
echo   ROYALTON RESORT AI SCREENER - WEB INTERFACE
echo   Enhanced with OCR and Semantic Matching
echo ===============================================
echo.

echo Starting Streamlit web interface...
echo Open your browser to: http://localhost:8501
echo.

echo To install enhanced features first time:
echo pip install -r enhanced_requirements.txt
echo python -m spacy download en_core_web_sm
echo.

cd /d "%~dp0"
streamlit run streamlit_gui.py

pause
