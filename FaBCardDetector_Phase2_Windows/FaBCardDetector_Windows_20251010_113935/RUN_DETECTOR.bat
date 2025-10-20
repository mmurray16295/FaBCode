@echo off
echo ========================================
echo FaB Card Detector - Starting...
echo ========================================
echo.

python fab_detector_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start!
    echo Make sure you ran INSTALL_WINDOWS.bat first.
    pause
)
