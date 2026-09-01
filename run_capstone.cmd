@echo off
setlocal
cd /d "%~dp0"

set "CAPSTONE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%CAPSTONE_PYTHON%" (
    echo The project virtual environment was not found.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting DeFaketive at http://127.0.0.1:5000
echo Keep this window open. Press Ctrl+C to stop the app.
start "" /b cmd /c "timeout /t 2 /nobreak ^>nul ^& start http://127.0.0.1:5000"
"%CAPSTONE_PYTHON%" webapp.py

if errorlevel 1 (
    echo.
    echo DeFaketive stopped with an error.
    pause
)
