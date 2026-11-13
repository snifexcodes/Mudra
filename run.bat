@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :nopython
)
rem "python -m pip" instead of pip.exe: Smart App Control blocks the unsigned pip.exe launcher
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
.venv\Scripts\python.exe app.py
pause
goto :eof
:nopython
echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add python.exe to PATH".
pause
