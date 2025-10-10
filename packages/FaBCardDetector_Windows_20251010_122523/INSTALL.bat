@echo off
REM Launch GUI installer without console window

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    REM Python not installed, need to show console for initial setup
    echo Python is not installed. Launching installer...
    echo.
    pythonw GUI_INSTALLER.py
    if errorlevel 1 (
        REM pythonw failed, try with python
        python GUI_INSTALLER.py
    )
) else (
    REM Python is installed, use pythonw to hide console
    start /B pythonw GUI_INSTALLER.py
    if errorlevel 1 (
        REM pythonw not available, fall back to python
        python GUI_INSTALLER.py
    )
)
