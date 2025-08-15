@echo off
title Royalton Resort - AI Resume Screener
cls
echo ================================================================
echo       ROYALTON RESORT - AI HIRING ASSISTANT
echo         One-Click Smart Candidate Selection
echo ================================================================
echo.
echo Installing/updating packages (if needed)...
pip install --quiet --upgrade pandas openpyxl "pdfminer.six>=20231228" python-docx 2>nul

echo.
echo Starting AI Resume Screener...
python royalton_ai_screener.py

if errorlevel 1 (
    echo.
    echo ERROR: Something went wrong!
    echo.
    echo Make sure you have Python installed:
    echo - Download from: https://python.org/downloads/
    echo - During installation, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

pause
