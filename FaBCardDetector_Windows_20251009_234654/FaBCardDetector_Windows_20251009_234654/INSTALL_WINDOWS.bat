@echo off
REM ============================================================
REM FaB Card Detector - One-Click Installer for Windows
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    FaB Card Detector - One-Click Setup
echo ============================================================
echo.

REM Get the directory where this script is located
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"

echo [1/5] Checking Python installation...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found! Installing Python...
    echo.
    
    REM Download Python installer (embedded version - lightweight!)
    echo Downloading Python 3.11 (embedded, ~20MB)...
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip' -OutFile 'python.zip'}"
    
    echo Extracting Python...
    powershell -Command "Expand-Archive -Path 'python.zip' -DestinationPath 'python' -Force"
    del python.zip
    
    REM Get pip for embedded Python
    echo Setting up pip...
    cd python
    echo import sys > ..\._pth
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'}"
    python.exe get-pip.py
    del get-pip.py
    cd ..
    
    set "PYTHON_CMD=%INSTALL_DIR%python\python.exe"
    set "PIP_CMD=%PYTHON_CMD% -m pip"
    
    echo Python installed successfully!
) else (
    echo Python found: 
    python --version
    set "PYTHON_CMD=python"
    set "PIP_CMD=python -m pip"
)

echo.
echo [2/5] Checking/Installing required packages (this may take 2-3 minutes)...
echo.

REM Check if packages are already installed
%PYTHON_CMD% -c "import torch; import cv2; import ultralytics; import mss" >nul 2>&1
if %errorlevel% equ 0 (
    echo All required packages already installed!
    echo Skipping package installation...
) else (
    echo Installing required packages...
    echo.
    
    REM Install dependencies
    echo Installing PyTorch (CPU version - lightweight)...
    %PIP_CMD% install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-warn-script-location
    
    echo Installing computer vision libraries...
    %PIP_CMD% install opencv-python-headless --no-warn-script-location
    
    echo Installing YOLO detector...
    %PIP_CMD% install ultralytics --no-warn-script-location
    
    echo Installing screen capture...
    %PIP_CMD% install mss pillow --no-warn-script-location
    
    echo Installing utilities...
    %PIP_CMD% install pyyaml requests numpy --no-warn-script-location
    
    echo.
    echo Package installation complete!
)

echo.
echo [3/5] Verifying installation...
echo.

REM Check if model exists
if not exist "models\best.pt" (
    echo WARNING: Model file not found at models\best.pt
    echo Please make sure best.pt is in the models folder.
    pause
)

REM Check if card data exists
if not exist "data\card.json" (
    echo WARNING: Card data not found at data\card.json
    echo Please make sure card.json is in the data folder.
    pause
)

echo Installation verified!

echo.
echo [4/5] Creating desktop shortcut...
echo.

REM Create a launcher script
echo @echo off > run_detector.bat
echo cd /d "%INSTALL_DIR%" >> run_detector.bat
echo %PYTHON_CMD% fab_detector_app.py >> run_detector.bat

REM Create desktop shortcut using PowerShell
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\FaB Card Detector.lnk"
powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%SHORTCUT_PATH%'); $Shortcut.TargetPath = '%INSTALL_DIR%run_detector.bat'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'FaB Card Detector'; $Shortcut.Save()"

echo Desktop shortcut created!

echo.
echo [5/5] Launching FaB Card Detector...
echo.

REM Launch the application with error checking
echo Starting detector...
%PYTHON_CMD% fab_detector_app.py

REM If we get here, the app closed or failed
if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo    ERROR: Failed to launch detector!
    echo ============================================================
    echo.
    echo Error code: %errorlevel%
    echo.
    echo Please check:
    echo   1. models\best.pt exists
    echo   2. data\card.json exists
    echo   3. All packages installed correctly
    echo.
    echo Try running RUN_DETECTOR.bat for more details.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    Setup Complete!
echo ============================================================
echo.
echo Desktop shortcut created: "FaB Card Detector"
echo To run again, double-click the shortcut or RUN_DETECTOR.bat
echo.
echo Press any key to close this window...
pause >nul
