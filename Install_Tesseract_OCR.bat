@echo off
echo ====================================
echo Installing Tesseract OCR for Windows
echo ====================================
echo.

echo 📋 This script will help you install Tesseract OCR engine
echo.
echo Option 1: Download and install Tesseract manually
echo Option 2: Install via Chocolatey (if available)
echo.

choice /c 12 /m "Choose installation method (1 or 2)"
if errorlevel 2 goto chocolatey
if errorlevel 1 goto manual

:manual
echo.
echo 🌐 Opening Tesseract download page...
echo Download the Windows installer from:
echo https://github.com/UB-Mannheim/tesseract/wiki
echo.
echo 📝 Installation Notes:
echo 1. Download the latest Windows installer (64-bit recommended)
echo 2. Run the installer as Administrator
echo 3. During installation, make sure to check "Add to PATH"
echo 4. Default installation path: C:\Program Files\Tesseract-OCR
echo.
echo After installation, restart your command prompt and run:
echo tesseract --version
echo.
start https://github.com/UB-Mannheim/tesseract/wiki
goto end

:chocolatey
echo.
echo 🍫 Attempting Chocolatey installation...
choco --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Chocolatey not found. Please install Chocolatey first:
    echo https://chocolatey.org/install
    start https://chocolatey.org/install
    goto end
)

echo Installing Tesseract via Chocolatey...
choco install tesseract -y
if errorlevel 1 (
    echo ❌ Chocolatey installation failed
    goto manual
) else (
    echo ✅ Tesseract installed successfully via Chocolatey
    echo.
    echo Testing installation...
    tesseract --version
    if errorlevel 1 (
        echo ⚠ Tesseract may need to be added to PATH manually
    ) else (
        echo ✅ Tesseract is working correctly!
    )
)
goto end

:end
echo.
echo 🔧 After Tesseract is installed, test the OCR functionality:
echo python test_ocr.py
echo.
echo If you get "tesseract is not installed or it's not in your PATH":
echo 1. Restart your command prompt
echo 2. Check that tesseract.exe is in your PATH
echo 3. Add C:\Program Files\Tesseract-OCR to your PATH if needed
echo.
pause
