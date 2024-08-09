::@echo off
echo Starting Ai

:: Esegui installDependencies.bat e, se ha successo, esegui python index.py
call script\installDependencies.bat
if %errorlevel% equ 0 (
    python index.py
) else (
    echo Failed to install dependencies.
    exit /b 1
)
