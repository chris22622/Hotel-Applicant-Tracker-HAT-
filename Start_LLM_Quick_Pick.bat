@echo off
echo ===========================================
echo   HAT - LLM Quick Pick
echo ===========================================
echo.

:: Ensure virtual environment
if not exist ".venv" (
    echo 🔧 Setting up Python environment...
    call Quick_Setup.bat
    if errorlevel 1 (
        echo ERROR: Setup failed
        pause
        exit /b 1
    )
)

:: Activate venv
call .venv\Scripts\activate.bat

echo.
echo Enter the position title you are hiring for:
set /p POSITION="Position: "

echo Enter how many candidates you want returned:
set /p TOPN="Top N: "

echo.
echo Using ChatGPT to re-rank top candidates by default.
echo (If you want to disable it for this run, type N and press Enter.)
set /p USELLM="Use LLM [Y/n]: "

set LLM_FLAG=--llm
if /I "%USELLM%"=="N" set LLM_FLAG=

echo.
echo Running quick pick for "%POSITION%"; Top %TOPN% %USELLM%
python llm_quick_pick.py -p "%POSITION%" -t %TOPN% %LLM_FLAG%

echo.
echo ✅ Done. Check the screening_results folder for files.
pause
