@echo off
REM Run the application without showing console window

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not installed!
    echo Please run INSTALL.bat first
    pause
    exit /b 1
)

REM Use pythonw to run without console window
start /B pythonw fab_detector_app.py
if errorlevel 1 (
    REM pythonw failed, try with regular python
    python fab_detector_app.py
)
