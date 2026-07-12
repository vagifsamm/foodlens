@echo off
rem FoodLens demo - iki defe klikle, brauzerde acilacaq.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [XETA] .venv tapilmadi. Evvelce PowerShell-de bunu isledin:  .\run.ps1 install
    pause
    exit /b 1
)
echo FoodLens demo baslayir... Brauzer avtomatik acilacaq (http://localhost:8501)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
