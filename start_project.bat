@echo off
cd /d "D:\Project trial\Emotion based project"

REM Create virtual environment if not already created
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call "venv\Scripts\activate.bat"

REM Install required packages
echo Installing packages...
pip install --upgrade pip
pip install flask pandas textblob

REM Run the Flask app
echo Starting Flask app...
python app.py

pause
