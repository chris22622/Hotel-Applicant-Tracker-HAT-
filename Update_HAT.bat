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

:: Detect if this is a Git working copy; if not, attempt conversion (zip → git)
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 goto :convert_to_git

:: Show current branch/status
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

:convert_to_git
echo ⚠️  This folder is not a Git repository.
echo     We'll convert it to a Git working copy so future updates are seamless.
set REPO_URL=https://github.com/chris22622/Hotel-Applicant-Tracker-HAT-.git
set TMP_ROOT=%TEMP%\hat_update_%RANDOM%%RANDOM%
set TMP_REPO=%TMP_ROOT%\repo
set BACKUP_DIR=%CD%\backups\update_%DATE:~10,4%%DATE:~4,2%%DATE:~7,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%

:: Normalize time with no spaces/colons
set BACKUP_DIR=%BACKUP_DIR: =0%
set BACKUP_DIR=%BACKUP_DIR::=%

echo.
echo Creating temporary workspace: %TMP_REPO%
mkdir "%TMP_REPO%" >nul 2>&1
if errorlevel 1 (
  echo ❌ Could not create temporary directory.
  goto :gitfail
)

echo ⬇️ Cloning latest repository snapshot...
git clone --depth 1 "%REPO_URL%" "%TMP_REPO%"
if errorlevel 1 (
  echo ❌ Failed to clone repository from %REPO_URL%.
  goto :gitfail
)

echo 💾 Backing up your local data to: %BACKUP_DIR%
mkdir "%BACKUP_DIR%" >nul 2>&1

:: Backup selected user data (best-effort)
if exist "config" robocopy "config" "%BACKUP_DIR%\config" /E /NFL /NDL /NJH /NJS >nul
if exist "hr_ats.db" copy /Y "hr_ats.db" "%BACKUP_DIR%\hr_ats.db" >nul
if exist "input_resumes" robocopy "input_resumes" "%BACKUP_DIR%\input_resumes" /E /NFL /NDL /NJH /NJS >nul
if exist "screening_results" robocopy "screening_results" "%BACKUP_DIR%\screening_results" /E /NFL /NDL /NJH /NJS >nul
if exist "var" robocopy "var" "%BACKUP_DIR%\var" /E /NFL /NDL /NJH /NJS >nul
if exist "plugins" robocopy "plugins" "%BACKUP_DIR%\plugins" /E /NFL /NDL /NJH /NJS >nul

echo 🔁 Copying repository files into current folder (preserving your data)...
:: Copy all repo files except .git and common local-only folders
robocopy "%TMP_REPO%" "%CD%" /E /NFL /NDL /NJH /NJS /XD ".git" ".venv" "input_resumes" "screening_results" "var" /XF "config\openai_api_key.txt" "config\secrets.yaml" >nul

:: Ensure .git metadata is present so future updates work with git
if not exist ".git" (
  echo 🧩 Installing Git metadata...
  robocopy "%TMP_REPO%\.git" ".git" /E /NFL /NDL /NJH /NJS >nul
)

:: Restore secrets if they exist in backup
if exist "%BACKUP_DIR%\config\openai_api_key.txt" copy /Y "%BACKUP_DIR%\config\openai_api_key.txt" "config\openai_api_key.txt" >nul
if exist "%BACKUP_DIR%\config\secrets.yaml" copy /Y "%BACKUP_DIR%\config\secrets.yaml" "config\secrets.yaml" >nul

echo ✅ Conversion complete. Proceeding with standard update...
echo.
goto :eof
