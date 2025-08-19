@echo off
echo ===========================================
echo   Hotel AI Resume Screener - CLI Mode
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
call .venv\Scripts\activate.bat

echo Available positions:
echo 1. Front Desk Agent
echo 2. Guest Services Manager  
echo 3. Executive Chef
echo 4. Housekeeping Manager
echo 5. Maintenance Supervisor
echo 6. Custom position
echo.

set /p choice="Select position (1-6): "

set position=""
if "%choice%"=="1" set "position=Front Desk Agent"
if "%choice%"=="2" set "position=Guest Services Manager"
if "%choice%"=="3" set "position=Executive Chef"
if "%choice%"=="4" set "position=Housekeeping Manager"
if "%choice%"=="5" set "position=Maintenance Supervisor"
if "%choice%"=="6" (
    set /p position="Enter custom position name: "
)

if "%position%"=="" (
    echo Invalid selection
    pause
    exit /b 1
)

echo.
echo 🎯 Screening for position: %position%
echo 📁 Processing resumes from: input_resumes\
echo 📊 Results will be saved to: screening_results\
echo.

:: Run the enhanced screener
python enhanced_ai_screener.py --position "%position%" --input input_resumes --output screening_results --export-excel

echo.
echo ✅ Screening complete!
echo 📁 Check the 'screening_results' folder for your reports
echo.
pause
