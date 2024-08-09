@echo off
setlocal enabledelayedexpansion

:: Controlla se il file dependencies.txt esiste
if not exist script/dependencies.txt (
    echo The dependencies.txt file doesn't exist.
    exit /b 1
)

:: Legge il file dependencies.txt riga per riga
set "success=0"
for /f "tokens=*" %%i in (script/dependencies.txt) do (
    :: Controlla se il pacchetto è già installato
    set "package_installed=false"
    for /f "tokens=*" %%j in ('pip freeze') do (
        :: split the line by == and get the first token
        for /f "tokens=1 delims=^=" %%k in ("%%j") do (
            set "package=%%k"

            if "%%i"=="!package!" (
                set "package_installed=true"
            )
        )
    )
    
    if "!package_installed!"=="false" (
        echo Installing %%i
        pip install %%i
        if %errorlevel% neq 0 (
            echo Failed to install %%i
            set "success=1"
        )
    ) else (
        echo %%i already installed.
    )
)

:: Se c'è stato un errore durante l'installazione, imposta l'errorlevel
if %success% neq 0 (
    exit /b 1
)

echo All dependencies are installed.
exit /b 0
