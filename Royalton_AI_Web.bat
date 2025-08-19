@echo off
title Hotel AI Resume Screener - Web App
echo ===========================================
echo   Hotel AI Resume Screener - Web App
echo ===========================================
echo.

:: Check virtual environment
if not exist ".venv" (
    echo ⚠️  Setting up virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install required packages
echo 📦 Installing required packages...
pip install fastapi uvicorn sqlalchemy pydantic-settings --quiet --disable-pip-version-check

:: Create data directory
if not exist "data" mkdir data

echo.
echo 🚀 Starting the web server...
echo 🌐 Go to: http://localhost:8000
echo.

:: Start the application directly
python start_web_app.py

pause
