@echo off
echo ========================================
echo FaB Card Detector Phase 4 - Installation
echo ========================================
echo.
echo This will install all required Python packages.
echo Please ensure Python 3.10+ is installed and in PATH.
echo.
pause

echo.
echo Installing packages...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Run RUN.bat to start the application.
echo.
pause
