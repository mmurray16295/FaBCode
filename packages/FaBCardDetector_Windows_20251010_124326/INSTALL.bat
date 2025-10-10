@echo off
echo ========================================
echo FaB Card Detector - Installation
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

REM Show Python version
echo Checking Python version...
python --version
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

echo Installing required packages...
echo This may take 5-10 minutes...
echo.

python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo.
    echo Please check:
    echo 1. You have Python 3.8 or higher
    echo 2. You have an internet connection
    echo 3. Try running as Administrator
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Double-click RUN.bat to start the application.
echo.
pause
