Set-Location "D:\Claude\Agent Files\FoodLens"
# num_workers=2: nw=4 exhausted RAM commit on this 16GB machine and WDDM then
# fails GPU allocations with spurious OOM (measured; see DECISIONS.md).

& .\.venv\Scripts\python.exe -m src.cnn.train --model simple --epochs 15 --bs 32 --num-workers 2 *>> training.log
if ($LASTEXITCODE -eq 0) { Add-Content training.log "SIMPLE_DONE" } else { Add-Content training.log "SIMPLE_FAILED"; exit 1 }
& .\.venv\Scripts\python.exe -m src.cnn.train --model effnet --epochs 10 --bs 32 --num-workers 2 *>> training.log
if ($LASTEXITCODE -eq 0) { Add-Content training.log "EFFNET_DONE" } else { Add-Content training.log "EFFNET_FAILED"; exit 1 }
