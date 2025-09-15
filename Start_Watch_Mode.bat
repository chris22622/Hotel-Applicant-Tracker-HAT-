@echo off
setlocal ENABLEDELAYEDEXPANSION

echo ===========================================
echo   Hotel AI Screener - Watch Mode
echo ===========================================
echo.

set INPUT_DIR=input_resumes
set DEFAULT_EXT=pdf,doc,docx,txt

:: Ensure venv
if not exist ".venv" (
  echo 🔧 Setting up environment...
  call Quick_Setup.bat
  if errorlevel 1 (
    echo ERROR: Setup failed
    pause
    exit /b 1
  )
)

call .venv\Scripts\activate.bat

:: Prompt for position
set /p POSITION=Enter target position (e.g., Front Office Manager): 
if "%POSITION%"=="" (
  echo Position is required.
  pause
  exit /b 1
)

:: Prompt for interval
set /p INTERVAL=Polling interval seconds [5]: 
if "%INTERVAL%"=="" set INTERVAL=5

:: Prompt for explain mode
set /p EXPLAIN=Enable explain mode? (y/N): 
if /I "%EXPLAIN%"=="Y" (
  set EXPLAIN_FLAG=--explain
) else (
  set EXPLAIN_FLAG=
)

:: Prompt for file extensions (optional)
set /p EXT=Allowed file extensions comma-separated [%DEFAULT_EXT%]: 
if "%EXT%"=="" set EXT=%DEFAULT_EXT%

if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"

echo.
echo 📂 Watching folder: %INPUT_DIR%
echo 🎯 Position: %POSITION%
echo ⏱  Interval: %INTERVAL%s

echo 🛑 Press Ctrl+C to stop.
python watch_input.py --position "%POSITION%" --input "%INPUT_DIR%" --interval %INTERVAL% --ext %EXT% %EXPLAIN_FLAG%

echo.
echo ✅ Watch mode ended.
pause
