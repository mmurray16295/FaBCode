@echo off
REM Quick launcher for FaB Card Detector

cd /d "%~dp0"

REM Check if using embedded Python or system Python
if exist "python\python.exe" (
    python\python.exe fab_detector_app.py
) else (
    python fab_detector_app.py
)

if %errorlevel% neq 0 (
    echo.
    echo Error launching detector!
    echo Run INSTALL_WINDOWS.bat first if you haven't already.
    echo.
    pause
)
