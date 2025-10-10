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

REM Check Python version is 3.8+
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.8 or higher is required!
    echo Your Python version is too old.
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo See TROUBLESHOOTING.txt for help.
    echo.
    pause
    exit /b 1
)

echo Python version OK!
echo.

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)

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
    echo 3. Run this as Administrator
    echo.
    echo See TROUBLESHOOTING.txt for detailed help.
    echo Or run CHECK_SYSTEM.bat to diagnose issues.
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Run RUN_DETECTOR.bat to start the application.
echo Or run CHECK_SYSTEM.bat to verify installation.
echo.
pause
