@echo off
setlocal
cd /d "%~dp0"
title LLM cascade study - leave this window open
echo ============================================================
echo  LLM cascade study - real-data experiments (E1-E5)
echo  Leave this window open. Safe to close and re-run: it resumes.
echo ============================================================

set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY (python --version >nul 2>&1 && set "PY=python")
if not defined PY (
  echo.
  echo  Python 3.10+ is not installed.
  echo  Install it from https://www.python.org/downloads/  and tick
  echo  "Add python.exe to PATH" in the installer, then run start.bat again.
  pause & exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)" || (
  echo  Python 3.10 or newer is required. Found:& %PY% --version & pause & exit /b 1)

if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating private Python environment in .venv ...
  %PY% -m venv .venv || (echo venv creation failed & pause & exit /b 1)
)
echo [setup] installing packages (first run only takes a few minutes) ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || (
  echo  Package install failed - check the internet connection and run start.bat again.
  pause & exit /b 1)

set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore
".venv\Scripts\python.exe" src\run_study.py
echo.
echo  All done. Send back the results_*.zip file in this folder.
pause
