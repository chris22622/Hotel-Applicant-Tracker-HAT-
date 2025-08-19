@echo off
echo ===========================================
echo   Stopping Hotel AI Resume Screener
===========================================
echo.

echo 🛑 Stopping any running Streamlit applications...

:: Kill any existing Streamlit processes
taskkill /f /im streamlit.exe >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq streamlit*" >nul 2>&1

:: Find and kill processes using common Streamlit ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8501') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8502') do taskkill /f /pid %%a >nul 2>&1

echo ✅ All Streamlit processes stopped
echo.
echo You can now run Quick_Start_Screener.bat again
echo.
pause
