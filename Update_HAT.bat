@echo off
setlocal enableextensions enabledelayedexpansion
echo ===========================================
echo   Hotel AI Resume Screener - Updater
echo ===========================================
echo.

:: Ensure we're in repo root (where this script lives)
pushd "%~dp0"

:: Check Git
git --version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Git is not installed or not in PATH.
  echo Download: https://git-scm.com/download/win
  pause
  exit /b 1
)

:: Check internet connectivity (quick ping to github)
ping -n 1 github.com >nul 2>&1
if errorlevel 1 (
  echo ⚠️  No internet connection detected. Cannot update.
  goto :end
)

:: Show current branch
echo Checking repository status...
git status -sb

:: Stash local changes if any
for /f "tokens=*" %%i in ('git status --porcelain') do set HAS_CHANGES=1
if defined HAS_CHANGES (
  echo 💾 Stashing local changes...
  git stash push -u -m "auto-updater-%DATE%-%TIME%" >nul
)

:: Fetch and pull latest from origin/main
echo 🔄 Fetching latest changes...
git fetch origin
if errorlevel 1 goto :gitfail

echo ⬇️ Pulling updates...
git pull --rebase origin main
if errorlevel 1 goto :gitfail

:: Reactivate venv and update deps
if exist .venv\Scripts\activate.bat (
  echo 🔧 Activating virtual environment...
  call .venv\Scripts\activate.bat
  echo 📦 Upgrading pip...
  python -m pip install --upgrade "pip<24.1" --quiet
  echo 📦 Syncing core dependencies...
  python -m pip install -r requirements.txt --quiet 2>nul
  if exist enhanced_requirements.txt (
    echo 📦 Syncing enhanced dependencies...
    python -m pip install -r enhanced_requirements.txt --quiet 2>nul
  )
  echo 📦 Ensuring OpenAI SDK installed...
  python -m pip install "openai>=1.40.0" --quiet
) else (
  echo ⚠️  No virtual environment detected. Run Quick_Setup.bat first.
)

:: If we stashed changes, try to re-apply them
if defined HAS_CHANGES (
  echo ♻️  Restoring local changes...
  git stash pop
  if errorlevel 1 (
    echo ⚠️  Could not automatically re-apply stashed changes. They remain in your stash.
    echo     Run 'git stash list' to view them.
  )
)

echo ✅ Update complete.
echo.
echo Next steps:
echo - Run Quick_Start_Screener.bat to launch
echo - If issues occur, try deleting .venv and running Quick_Setup.bat

:end
popd
endlocal
pause
:gitfail
echo ❌ Git operation failed. Please check your network or repository status.
popd
endlocal
pause
