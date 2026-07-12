@echo off
rem ============================================================
rem  FoodLens - HAMISI BIR KLIKLE
rem  Bu fayl her seyi ozu isledir: yoxlayir, API-ni acir,
rem  demonu acir, brauzeri acir. Manual hecne lazim deyil.
rem ============================================================
cd /d "%~dp0"
title FoodLens Launcher
echo.
echo  ============================================
echo   FoodLens baslayir...
echo  ============================================
echo.

rem --- 1. venv yoxlanisi (yoxdursa avtomatik qurulur) ---
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Virtual muhit qurulur, ilk defe 5-10 deq ceke biler...
    python -m venv .venv
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
    echo [1/4] Virtual muhit OK
)

rem --- 2. data yoxlanisi ---
if not exist "data\nutrition_db.json" (
    echo [2/4] Nutrition DB qurulur...
    ".venv\Scripts\python.exe" scripts\build_nutrition_db.py
) else (
    echo [2/4] Nutrition DB OK
)

rem --- 3. model yoxlanisi (yalniz xeberdarliq) ---
if not exist "models\effnet_best.pt" (
    echo [3/4] XEBERDARLIQ: model oyredilmeyib, Sekil analizi tabi islemeyecek.
    echo        Metn tabi ve Gundelik iseyecek.
) else (
    echo [3/4] Model OK: models\effnet_best.pt
)

rem --- 4. API + Demo ---
echo [4/4] API ve Demo acilir...
start "FoodLens API" /min cmd /c ""%~dp0.venv\Scripts\python.exe" -m uvicorn src.api:app --host 127.0.0.1 --port 8000"
start "" http://127.0.0.1:8000/docs
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py

rem Streamlit baglananda API penceresini de bagla
taskkill /fi "WINDOWTITLE eq FoodLens API*" /f >nul 2>&1
