@echo off
REM ============================================================
REM  PCB Defect Detection - Backend (FastAPI)
REM  Runs uvicorn inside the project's pipenv environment.
REM ============================================================

REM Move to this script's folder (project root, where Pipfile lives)
cd /d "%~dp0"

echo(
echo Starting BACKEND (FastAPI) at http://127.0.0.1:8000
echo   Interactive docs: http://127.0.0.1:8000/docs
echo   First startup takes ~30s (loads YOLO model + ChromaDB).
echo   Press Ctrl+C in this window to stop.
echo(

pipenv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

echo(
echo Backend stopped.
pause
