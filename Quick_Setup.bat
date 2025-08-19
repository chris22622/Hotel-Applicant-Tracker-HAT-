@echo off
echo ===========================================
echo   Hotel AI Resume Screener - Quick Setup
echo ===========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment exists
)

:: Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

:: Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip --quiet

:: Install core requirements
echo 📦 Installing core requirements...
python -m pip install pandas openpyxl pyyaml streamlit plotly matplotlib seaborn --quiet

:: Install enhanced AI packages (optional)
echo 📦 Installing AI enhancement packages...
python -m pip install spacy scikit-learn textblob --quiet
if not errorlevel 1 (
    echo 📦 Downloading spaCy language model...
    python -m spacy download en_core_web_sm --quiet
    if not errorlevel 1 (
        echo ✅ AI enhancements installed
    ) else (
        echo ⚠️  SpaCy model download failed - basic functionality will work
    )
) else (
    echo ⚠️  AI enhancements failed - basic functionality will work
)

:: Install OCR packages (optional)
echo 📦 Installing OCR packages...
python -m pip install pytesseract pdf2image PyPDF2 python-docx pillow --quiet
if not errorlevel 1 (
    echo ✅ OCR support installed
    echo ℹ️  Note: You may need to install Tesseract OCR separately for image processing
    echo ℹ️  Download from: https://github.com/UB-Mannheim/tesseract/wiki
) else (
    echo ⚠️  OCR packages failed - text files will work
)

:: Create input and output directories
if not exist "input_resumes" (
    mkdir input_resumes
    echo ✅ Created input_resumes directory
)

if not exist "screening_results" (
    mkdir screening_results
    echo ✅ Created screening_results directory
)

:: Create sample configuration
if not exist "enhanced_config.yaml" (
    echo 📝 Creating sample configuration...
    (
        echo positions:
        echo   Front_Desk_Agent:
        echo     description: "Front desk operations and guest services"
        echo     priority_skills: ["customer service", "PMS systems", "communication"]
        echo     min_experience: 0
        echo     preferred_experience: 2
        echo   Guest_Services_Manager:
        echo     description: "Guest services team leadership"
        echo     priority_skills: ["leadership", "guest relations", "team management"]
        echo     min_experience: 2
        echo     preferred_experience: 5
        echo   Executive_Chef:
        echo     description: "Kitchen operations and culinary leadership"
        echo     priority_skills: ["culinary arts", "kitchen management", "food safety"]
        echo     min_experience: 5
        echo     preferred_experience: 8
        echo.
        echo screening:
        echo   max_candidates: 50
        echo   score_threshold: 0.5
        echo   export_excel: true
    ) > enhanced_config.yaml
    echo ✅ Configuration file created
)

echo.
echo ===========================================
echo   🎉 SETUP COMPLETE!
echo ===========================================
echo.
echo Next steps:
echo 1. Place resume files in the 'input_resumes' folder
echo 2. Run 'Quick_Start_Screener.bat' to launch the web interface
echo 3. Or run 'Enhanced_CLI_Screener.bat' for command-line usage
echo.
echo For OCR support (scanned PDFs/images):
echo - Download Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
echo - Add it to your system PATH
echo.
pause
