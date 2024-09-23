@echo off

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python is already installed
) else (
    echo Python is not installed.
    echo Downloading Python installer...
    
    REM Download Python installer (change the version link if needed)
    powershell -Command "Start-BitsTransfer -Source https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe -Destination python-installer.exe"
    
    REM Run the installer silently (adjust options as needed)
    echo Installing Python...
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1
    
    REM Clean up installer
    del python-installer.exe
    
    echo Python installed successfully
)

pause

call script\installDependencies.bat
if %errorlevel% equ 0 (
    echo Dependencies installed successfully.
    echo Now you can run the program by typing "./start.bat" in the terminal.
) else (
    echo Failed to install dependencies.
    exit /b 1
)