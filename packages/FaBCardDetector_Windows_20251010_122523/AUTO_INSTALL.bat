@echo off
REM This batch file launches the PowerShell installer

echo ========================================
echo FaB Card Detector - Automatic Installer
echo ========================================
echo.
echo This will automatically:
echo - Install Python 3.11 if needed
echo - Upgrade pip
echo - Install all required packages
echo.
echo Note: This requires Administrator privileges
echo.
pause

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo.
    echo ERROR: This script must be run as Administrator
    echo.
    echo Right-click this file and select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

REM Run PowerShell script
echo.
echo Starting installer...
echo.

powershell.exe -ExecutionPolicy Bypass -File "%~dp0AUTO_INSTALL.ps1"

if errorlevel 1 (
    echo.
    echo Installation failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo Done!
pause
