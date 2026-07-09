@echo off
REM ============================================================
REM  PCB Defect Detection - Frontend (Streamlit GUI)
REM  Runs streamlit inside the project's pipenv environment.
REM  Start the backend first (run_backend.bat) so the sidebar
REM  shows "Model loaded".
REM ============================================================

REM Move to this script's folder (project root, where Pipfile lives)
cd /d "%~dp0"

echo(
echo Starting FRONTEND (Streamlit) at http://localhost:8501
echo   Make sure the backend is running (run_backend.bat).
echo   Press Ctrl+C in this window to stop.
echo(

pipenv run streamlit run src/gui/app.py --server.port 8501

echo(
echo Frontend stopped.
pause
