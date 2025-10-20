@echo off
echo ========================================
echo FaB Card Detector - Installation
echo ========================================
echo.
echo This will install required Python packages.
echo Make sure you have Python 3.8+ installed!
echo.
pause

python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo.
echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Run RUN_DETECTOR.bat to start the application.
pause
