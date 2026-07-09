@echo off
REM ============================================================
REM  PCB Defect Detection - Full App Launcher
REM  Opens the backend (FastAPI) and frontend (Streamlit) in
REM  two separate windows, both inside the project's pipenv env.
REM ============================================================

REM Move to this script's folder (project root, where Pipfile lives)
cd /d "%~dp0"

echo(
echo === PCB Defect Detection - starting both servers ===
echo(

echo [1/2] Launching BACKEND (FastAPI) on http://127.0.0.1:8000 ...
start "PCB Backend (API)" cmd /k pipenv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

echo       Waiting ~40s for the model + ChromaDB to load before starting the GUI...
echo       (If the GUI still says "Cannot reach API", just refresh the browser tab once the backend window shows "Application startup complete".)
timeout /t 40 /nobreak >nul

echo [2/2] Launching FRONTEND (Streamlit) on http://localhost:8501 ...
start "PCB Frontend (GUI)" cmd /k pipenv run streamlit run src/gui/app.py --server.port 8501

echo(
echo === Both servers launched in separate windows ===
echo   Backend : http://127.0.0.1:8000/docs
echo   Frontend: http://localhost:8501
echo(
echo Close either window (or press Ctrl+C in it) to stop that server.
echo This launcher window can be closed now.
echo(
pause
