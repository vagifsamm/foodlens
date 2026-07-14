Set-Location "D:\Claude\Agent Files\FoodLens"
& .\.venv\Scripts\python.exe -m src.cnn.train --model simple --epochs 15 --bs 64 --num-workers 4 *>> training.log
Add-Content training.log "SIMPLE_DONE"
& .\.venv\Scripts\python.exe -m src.cnn.train --model effnet --epochs 10 --bs 32 --num-workers 4 *>> training.log
Add-Content training.log "EFFNET_DONE"
