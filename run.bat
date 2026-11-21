@echo off
REM Owner: Niti. One command to launch the console on Windows:  run.bat   (run.bat --rebuild after git pull)
REM First run: creates .venv (py launcher, 3.12 then 3.11), installs requirements, then builds the data split,
REM S2's triage model and S3's retrieval index via scripts\prepare.py if they are missing.
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

set REBUILD=0
if "%~1"=="--rebuild" set REBUILD=1
set NEED=0
if "%REBUILD%"=="1" set NEED=1
if not exist models\triage.joblib set NEED=1
.venv\Scripts\python -c "from src.retrieval import index_ready; raise SystemExit(0 if index_ready() else 1)" 2>nul || set NEED=1
if "%NEED%"=="1" (
  echo Preparing data split, triage model and retrieval index ^(first run: several minutes^)...
  .venv\Scripts\python scripts\prepare.py --split --train --index || echo WARNING: setup incomplete; the console will use labelled stand-ins
)

if "%REBUILD%"=="1" (
  .venv\Scripts\python -m streamlit run app.py
) else (
  .venv\Scripts\python -m streamlit run app.py %*
)
