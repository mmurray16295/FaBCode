# FaB Card Detector - Automatic Installer with Python Setup
# This script will:
# 1. Check if Python is installed
# 2. Check Python version
# 3. Download and install Python 3.11 if needed
# 4. Install required packages

$ErrorActionPreference = "Stop"

# Configuration
$RequiredPythonMajor = 3
$RequiredPythonMinor = 8
$PythonDownloadVersion = "3.11.9"
$PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonDownloadVersion/python-$PythonDownloadVersion-amd64.exe"
$PythonInstallerPath = "$env:TEMP\python-installer.exe"

Write-Host "========================================"
Write-Host "FaB Card Detector - Automatic Installer"
Write-Host "========================================"
Write-Host ""

# Function to check if Python is installed and get version
function Get-PythonVersion {
    try {
        $version = & python --version 2>&1 | Out-String
        if ($version -match "Python (\d+)\.(\d+)\.(\d+)") {
            return @{
                Installed = $true
                Major = [int]$matches[1]
                Minor = [int]$matches[2]
                Patch = [int]$matches[3]
                FullVersion = "$($matches[1]).$($matches[2]).$($matches[3])"
            }
        }
    } catch {
        return @{ Installed = $false }
    }
    return @{ Installed = $false }
}

# Check current Python installation
Write-Host "Checking Python installation..." -ForegroundColor Cyan
$pythonInfo = Get-PythonVersion

if ($pythonInfo.Installed) {
    Write-Host "Python $($pythonInfo.FullVersion) detected" -ForegroundColor Green
    
    # Check if version meets requirements
    if (($pythonInfo.Major -gt $RequiredPythonMajor) -or 
        (($pythonInfo.Major -eq $RequiredPythonMajor) -and ($pythonInfo.Minor -ge $RequiredPythonMinor))) {
        Write-Host "Python version is compatible (3.8+ required)" -ForegroundColor Green
        $needsInstall = $false
    } else {
        Write-Host "Python version is too old (3.8+ required)" -ForegroundColor Yellow
        Write-Host "Need to install Python $PythonDownloadVersion" -ForegroundColor Yellow
        $needsInstall = $true
    }
} else {
    Write-Host "Python is not installed" -ForegroundColor Yellow
    Write-Host "Need to install Python $PythonDownloadVersion" -ForegroundColor Yellow
    $needsInstall = $true
}

# Install Python if needed
if ($needsInstall) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Installing Python $PythonDownloadVersion"
    Write-Host "========================================"
    Write-Host ""
    
    # Ask for confirmation
    $confirm = Read-Host "Do you want to install Python $PythonDownloadVersion? (Y/N)"
    if ($confirm -ne "Y" -and $confirm -ne "y") {
        Write-Host "Installation cancelled by user" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Download Python installer
    Write-Host "Downloading Python installer..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $PythonInstallerUrl -OutFile $PythonInstallerPath -UseBasicParsing
        Write-Host "Download complete!" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to download Python installer" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host ""
        Write-Host "Please download and install Python manually from:"
        Write-Host "https://www.python.org/downloads/"
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Install Python silently
    Write-Host ""
    Write-Host "Installing Python..." -ForegroundColor Cyan
    Write-Host "This may take a few minutes..." -ForegroundColor Cyan
    Write-Host ""
    
    try {
        # Silent install with all features
        $installArgs = @(
            "/quiet",
            "InstallAllUsers=1",
            "PrependPath=1",
            "Include_test=0",
            "Include_pip=1",
            "Include_doc=0"
        )
        
        $process = Start-Process -FilePath $PythonInstallerPath -ArgumentList $installArgs -Wait -PassThru
        
        if ($process.ExitCode -eq 0) {
            Write-Host "Python installed successfully!" -ForegroundColor Green
            
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            
            # Clean up installer
            Remove-Item $PythonInstallerPath -Force
            
            # Verify installation
            Start-Sleep -Seconds 2
            $pythonInfo = Get-PythonVersion
            if ($pythonInfo.Installed) {
                Write-Host "Verified: Python $($pythonInfo.FullVersion) is now installed" -ForegroundColor Green
            } else {
                Write-Host "WARNING: Python was installed but not detected in PATH" -ForegroundColor Yellow
                Write-Host "You may need to restart your computer" -ForegroundColor Yellow
            }
        } else {
            throw "Installer returned exit code $($process.ExitCode)"
        }
    } catch {
        Write-Host "ERROR: Failed to install Python" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        Write-Host ""
        Write-Host "Please try:"
        Write-Host "1. Run this script as Administrator (Right-click -> Run as Administrator)"
        Write-Host "2. Install Python manually from: https://www.python.org/downloads/"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Now install Python packages
Write-Host ""
Write-Host "========================================"
Write-Host "Installing Required Packages"
Write-Host "========================================"
Write-Host ""

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Cyan
try {
    & python -m pip install --upgrade pip --quiet
    Write-Host "Pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "WARNING: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing packages (this may take 5-10 minutes)..." -ForegroundColor Cyan
Write-Host ""

# Install from requirements.txt
try {
    & python -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "========================================"
        Write-Host "Installation Complete!"
        Write-Host "========================================"
        Write-Host ""
        Write-Host "All packages installed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "You can now run RUN_DETECTOR.bat to start the application." -ForegroundColor Cyan
        Write-Host "Or run CHECK_SYSTEM.bat to verify installation." -ForegroundColor Cyan
    } else {
        throw "pip install returned error code $LASTEXITCODE"
    }
} catch {
    Write-Host ""
    Write-Host "ERROR: Failed to install packages" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "1. You have an internet connection" -ForegroundColor Yellow
    Write-Host "2. Antivirus is not blocking pip" -ForegroundColor Yellow
    Write-Host "3. Try running as Administrator" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "See TROUBLESHOOTING.txt for detailed help" -ForegroundColor Cyan
    Write-Host "Or run CHECK_SYSTEM.bat to diagnose issues" -ForegroundColor Cyan
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Read-Host "Press Enter to close"
