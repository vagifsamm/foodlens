@echo off
rem Butun testleri isledir (pytest).
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest tests\ -v
pause
