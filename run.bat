@echo off
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || goto :nopython
)
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
python app.py
pause
goto :eof
:nopython
echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add python.exe to PATH".
pause
