@echo off
setlocal
cd /d "%~dp0"

"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\extract_waybill_rois.py" ^
  "G:\CQ\datasets\barcode" ^
  --model "%~dp0weights\best.pt" ^
  --output-dir "G:\CQ\datasets\barcode\barcode9.1" ^
  --conf 0.30 ^
  --padding 0.05 ^
  %*

endlocal
