@echo off
title Hotel AI Resume Screener - Web App (Auto-Open)
echo ===========================================
echo   Hotel AI Resume Screener - Web App
echo ===========================================
echo.

:: Check if virtual environment exists
if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install dependencies
echo 📦 Installing dependencies...
pip install fastapi uvicorn sqlalchemy pydantic-settings --quiet

:: Create data directory
if not exist "data" mkdir data

echo.
echo 🚀 Starting web server...
echo ⏳ Please wait 5 seconds for server to start...
echo.

:: Start the web server in background
start /B python start_web_app.py

:: Wait for server to start
timeout /t 5 >nul

:: Open browser automatically
echo 🌐 Opening web browser...
start "" http://localhost:8000

echo.
echo ✅ SUCCESS! The web application should now be open in your browser
echo.
echo 🌐 Main App: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo 🛑 To stop the server, close this window or press Ctrl+C
echo.
pause
