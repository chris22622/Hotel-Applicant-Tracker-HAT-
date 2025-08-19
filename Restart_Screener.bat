@echo off
echo ===========================================
echo   Restarting Hotel AI Resume Screener
===========================================
echo.

echo 🔄 Restarting the application to apply latest fixes...
echo.

:: Stop any running instances
call Stop_Screener.bat

:: Wait a moment
timeout /t 2 >nul

:: Start fresh
call Quick_Start_Screener.bat
