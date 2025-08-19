@echo off
echo ===========================================
echo   Hotel AI Resume Screener - Quick Start
echo ===========================================
echo.

:: Check if virtual environment exists
if not exist ".venv" (
    echo 🔧 Virtual environment not found. Running setup first...
    call Quick_Setup.bat
    if errorlevel 1 (
        echo ERROR: Setup failed
        pause
        exit /b 1
    )
    echo.
)

:: Activate virtual environment
echo 🚀 Starting Hotel AI Resume Screener...
call .venv\Scripts\activate.bat

:: Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing Streamlit...
    pip install streamlit --quiet
)

:: Start the enhanced Streamlit application
echo 🌐 Launching web interface...
echo.
echo ✨ The Hotel AI Resume Screener will open in your web browser
echo 📱 If it doesn't open automatically, go to: http://localhost:8501
echo 🛑 Press Ctrl+C in this window to stop the application
echo.

streamlit run enhanced_streamlit_app.py --server.port 8501 --server.headless false

pause
