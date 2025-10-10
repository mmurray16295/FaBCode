@echo off
echo Running system check...
python CHECK_SYSTEM.py
if errorlevel 1 (
    echo.
    echo Python check failed!
    echo Please run AUTO_INSTALL.bat as Administrator to install Python.
    pause
)
