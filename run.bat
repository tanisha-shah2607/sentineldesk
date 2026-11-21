@echo off
REM Owner: Niti. One command to launch the console on Windows:  run.bat
REM Creates .venv and installs requirements on first run. Uses the py launcher (3.12, then 3.11).
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  py -3.12 -m venv .venv 2>nul || py -3.11 -m venv .venv 2>nul || python -m venv .venv
  if errorlevel 1 (
    echo Python 3.11/3.12 not found
    exit /b 1
  )
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt
)
if not exist .env copy .env.example .env >nul

.venv\Scripts\python -m streamlit run app.py %*
