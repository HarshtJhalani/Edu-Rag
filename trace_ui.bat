@echo off
rem Opens the MLflow trace viewer for Edu-RAG at http://127.0.0.1:5000 (traces are stored in data\mlflow on this computer).
cd /d "%~dp0"
set "VIRTUAL_ENV=%~dp0rp"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
start "" http://127.0.0.1:5000
python -m mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db --host 127.0.0.1 --port 5000
pause
