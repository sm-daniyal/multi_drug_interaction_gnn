@echo off
title Multi-Drug Interaction Prediction System
echo =====================================================================
echo  Multi-Drug Interaction Prediction System (JNNCE BIS786 - Batch 25)
echo  Launching Clinical Decision Support System...
echo =====================================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [NOTICE] Virtual environment not found. Running first-time setup...
    call INSTALL_AND_RUN.bat
    exit /b 0
)

call .\venv\Scripts\activate
echo Opening in default browser at http://localhost:8501
echo (Press Ctrl+C in this window anytime to stop the server)
echo.
streamlit run app/app.py
pause
