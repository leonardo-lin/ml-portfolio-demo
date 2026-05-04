@echo off
echo ============================================================
echo  QLoRA Demo Site - Starting Streamlit App
echo ============================================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run install.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Virtual environment activated.

echo.
echo Starting Streamlit on http://localhost:8501
echo Press Ctrl+C to stop.
echo.

streamlit run app.py --server.port 8501

pause
