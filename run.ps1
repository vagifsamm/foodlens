# FoodLens task runner (Makefile equivalent for Windows).
# Usage: .\run.ps1 <target>   e.g.  .\run.ps1 train-effnet
param([Parameter(Position = 0)][string]$Target = "help", [Parameter(ValueFromRemainingArguments = $true)]$Rest)

$ErrorActionPreference = "Stop"
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Set-Location $PSScriptRoot

switch ($Target) {
    "install" {
        python -m venv .venv
        & $Py -m pip install -r requirements.txt
        & $Py -m pip freeze | Out-File -Encoding utf8 requirements.lock.txt
    }
    "data" {
        & $Py scripts/prepare_data.py
        & $Py scripts/build_nutrition_db.py
    }
    "train-simple" { & $Py -m src.cnn.train --model simple --epochs 15 --bs 64 @Rest }
    "train-effnet" { & $Py -m src.cnn.train --model effnet --epochs 10 --bs 32 --mixed-precision @Rest }
    "evaluate"     { & $Py -m src.cnn.evaluate @Rest }
    "api"          { & $Py -m uvicorn src.api:app --host 127.0.0.1 --port 8000 }
    "demo"         { & $Py -m streamlit run app/streamlit_app.py }
    "test"         { & $Py -m pytest tests/ -v }
    default {
        Write-Host "Targets: install | data | train-simple | train-effnet | evaluate | api | demo | test"
    }
}
