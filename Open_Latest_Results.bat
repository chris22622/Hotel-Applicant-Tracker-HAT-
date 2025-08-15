@echo off
echo ===============================================
echo   OPEN LATEST ROYALTON SCREENING RESULTS
echo   In Excel with all candidate contact info
echo ===============================================
echo.

echo Searching for latest screening results...
cd /d "%~dp0"

REM Find the most recent results folder
for /f "delims=" %%i in ('dir "screening_results" /b /ad /o-d 2^>nul') do (
    set "latest_folder=%%i"
    goto found_folder
)

echo ❌ No screening results found!
echo Run Royalton_AI_Screener.bat first to create results.
pause
exit /b 1

:found_folder
echo ✅ Found latest results: %latest_folder%

REM Find the Excel file in the latest folder
for /f "delims=" %%i in ('dir "screening_results\%latest_folder%\Royalton_Screening_Results*.xlsx" /b 2^>nul') do (
    set "excel_file=%%i"
    goto found_excel
)

echo ❌ No Excel file found in latest results!
echo The screening may have created CSV files instead.
echo Install: pip install xlsxwriter
pause
exit /b 1

:found_excel
echo ✅ Found Excel file: %excel_file%
echo.
echo Opening in Excel...
echo 📞 Go to "CANDIDATES TO CALL" sheet first!
echo.

start excel "screening_results\%latest_folder%\%excel_file%"

echo ✅ Excel should now be opening with your candidate contact list!
echo.
pause
