@echo off
rem Change directory to the script's location
cd /d %~dp0

echo Installing required packages...
pip install -r requirements.txt
pip install pyinstaller

echo Building executable...
rem Add the logo file to the bundle using --add-data
rem Source path is relative to this build script (in resume-matcher)
rem Destination path "." means the root directory inside the bundle.
rem Temporarily removed --icon=resume-icon.ico
pyinstaller --name ResumeRanker --onefile --windowed --add-data "public\images\ATOMS_LOGO.png;." resume_matcher_enhanced.py

echo Executable created in the 'dist' directory
pause 