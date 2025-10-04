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

:: Install OpenAI client for ChatGPT (recommended)
echo 📦 Installing ChatGPT client (OpenAI SDK)...
python -m pip install openai --quiet
if not errorlevel 1 (
    echo ✅ ChatGPT client installed
) else (
    echo ⚠️  ChatGPT client install failed - LLM features will be disabled
)

:: Optional helper for better DOCX extraction (safe with modern pip)
choice /C YN /N /M "Install extra text extractor (docx2txt)? [Y/N]: "
if errorlevel 2 goto skip_text_helpers
python -m pip install docx2txt --quiet
if not errorlevel 1 (
    echo ✅ docx2txt installed
    echo ℹ️  Note: Legacy .doc files are not fully supported without additional tools.
    echo ℹ️  This app will still try PDF text extraction, RTF stripping, or OCR fallbacks when possible.
) else (
    echo ⚠️  docx2txt install failed (optional)
)
:skip_text_helpers

:: Optional embeddings (sentence-transformers)
choice /C YN /N /M "Install embeddings package (sentence-transformers)? [Y/N]: "
if errorlevel 2 goto skip_embeddings
python -m pip install sentence-transformers --quiet
if not errorlevel 1 (
    echo ✅ Embeddings package installed
) else (
    echo ⚠️  Embeddings package install failed (optional)
)
:skip_embeddings

:: Prompt to set OPENAI_API_KEY persistently (Windows only)
echo.
echo Would you like to set your OpenAI API key now for LLM features?
echo This will store it in your user environment and also set it for this session.
choice /C YN /N /M "Set OPENAI_API_KEY now? [Y/N]: "
if errorlevel 2 goto skip_set_openai_key

set "OPENAI_API_KEY="
set /p OPENAI_API_KEY="Enter your OpenAI API key (starts with sk-): "
if "%OPENAI_API_KEY%"=="" (
    echo ⚠️  No key entered. You can set it later with:
    echo     setx OPENAI_API_KEY "sk-..."
    goto after_set_openai_key
)

echo 🔐 Storing OPENAI_API_KEY for current user (you may need to reopen terminals)...
setx OPENAI_API_KEY "%OPENAI_API_KEY%" >nul
if errorlevel 1 (
    echo ⚠️  Could not persist OPENAI_API_KEY. You can set it manually later with:
    echo     setx OPENAI_API_KEY "%OPENAI_API_KEY%"
) else (
    echo ✅ OPENAI_API_KEY saved. New shells will inherit it.
)

:: Also set it for the current session so you can use it immediately
set "OPENAI_API_KEY=%OPENAI_API_KEY%"
echo ✅ OPENAI_API_KEY available for this session.

:after_set_openai_key

:skip_set_openai_key

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
echo Press any key to close this window...
pause >nul
