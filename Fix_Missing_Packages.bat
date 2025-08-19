@echo off
echo ===========================================
echo   Hotel AI Resume Screener - Fix Setup
echo ===========================================
echo.

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install missing packages
echo 📦 Installing missing visualization packages...
pip install plotly matplotlib seaborn --quiet

echo 📦 Installing additional helpful packages...
pip install streamlit-aggrid tqdm textblob --quiet

echo ✅ All packages installed successfully!
echo.
echo You can now run:
echo - Quick_Start_Screener.bat (for web interface)
echo - Enhanced_CLI_Screener.bat (for command line)
echo.
pause
