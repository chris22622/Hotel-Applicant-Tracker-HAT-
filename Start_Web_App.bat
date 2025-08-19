@echo off
echo ===========================================
echo   Hotel AI Resume Screener - Web App
echo ===========================================
echo.

:: Activate virtual environment if it exists
if exist ".venv" (
    echo 🔄 Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo ⚠️  No virtual environment found. Run Quick_Setup.bat first.
    echo.
    echo 🔧 Creating virtual environment now...
    python -m venv .venv
    call .venv\Scripts\activate.bat
)

:: Install additional web dependencies if needed
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn, sqlalchemy" >nul 2>&1
if errorlevel 1 (
    echo 📦 Installing web application dependencies...
    pip install fastapi uvicorn sqlalchemy pydantic-settings --quiet
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies already installed
)

:: Create data directory
if not exist "data" mkdir data

:: Start the web application
echo 🌐 Starting Hotel AI Resume Screener Web Application...
echo.
echo ⏳ Please wait while the server starts...
echo.

:: Start the server in background and open browser
start "" python start_web_app.py

:: Wait a moment for server to start
timeout /t 3 >nul

:: Open the browser automatically
echo 🚀 Opening web browser...
start "" http://localhost:8000

echo.
echo ✅ Web application is starting!
echo 🌐 If browser doesn't open, go to: http://localhost:8000
echo 📚 API documentation: http://localhost:8000/docs
echo.
echo 🛑 To stop the server, close the Python console window that opened
echo.
pause
