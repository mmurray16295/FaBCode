@echo off
REM Quick launcher for FaB Card Detector

echo.
echo ============================================================
echo    FaB Card Detector - Starting...
echo ============================================================
echo.

cd /d "%~dp0"

REM Check if using embedded Python or system Python
if exist "python\python.exe" (
    echo Using embedded Python...
    python\python.exe fab_detector_app.py
) else (
    echo Using system Python...
    python fab_detector_app.py
)

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo    ERROR: Detector failed to launch!
    echo ============================================================
    echo.
    echo Error code: %errorlevel%
    echo.
    echo This could mean:
    echo   1. Python packages not installed - Run INSTALL_WINDOWS.bat
    echo   2. Model file missing - Check models\best.pt exists
    echo   3. Data files missing - Check data\card.json exists
    echo.
    echo If this is your first time, please run INSTALL_WINDOWS.bat
    echo.
    pause
    exit /b 1
)

echo.
echo Detector closed successfully.
echo.
