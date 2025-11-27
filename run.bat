@echo off
REM PromptFuzz CLI Launcher
REM Double-click this file to start the interactive menu

echo Starting PromptFuzz Interactive Mode...
echo.

REM Check if virtual environment exists
if not exist .venv (
    echo Error: Virtual environment not found!
    echo Please run setup first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -e .
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment and run
call .venv\Scripts\activate.bat
python src\cli.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to close...
    pause >nul
)
