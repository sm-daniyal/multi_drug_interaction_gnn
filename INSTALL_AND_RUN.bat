@echo off
title Multi-Drug Interaction Prediction System - Setup & Launch
echo =====================================================================
echo  Multi-Drug Interaction Prediction System (JNNCE BIS786 - Batch 25)
echo  Automated First-Time Setup & Launcher
echo =====================================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10 or higher from https://python.org
    echo Make sure to check the box "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Setting up Python virtual environment (venv)...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [2/3] Installing required packages from requirements.txt...
call .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo [WARNING] Some packages may have reported warnings during installation.
)

echo.
echo =====================================================================
echo [3/3] Launching Clinical Decision Support System...
echo Opening in default browser at http://localhost:8501
echo (Press Ctrl+C in this window anytime to stop the server)
echo =====================================================================
echo.
streamlit run app/app.py
pause
