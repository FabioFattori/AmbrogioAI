call script\installDependencies.bat
if %errorlevel% equ 0 (
    echo Dependencies installed successfully.
    echo Now you can run the program by typing "./start.bat" in the terminal.
) else (
    echo Failed to install dependencies.
    exit /b 1
)