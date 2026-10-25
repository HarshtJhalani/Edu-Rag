@echo off
rem Starts the Edu-RAG app. Activates the project venv relative to this file (rp\Scripts\activate.bat
rem still holds the folder's old path), then runs Streamlit.
cd /d "%~dp0"
set "VIRTUAL_ENV=%~dp0rp"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
python -m streamlit run app.py
pause
