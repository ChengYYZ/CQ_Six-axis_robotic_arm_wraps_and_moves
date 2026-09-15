@echo off
setlocal
cd /d "%~dp0"

"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\offline_waybill_test.py" ^
  "G:\CQ\datasets\shipping_label_2.0_8.31\test" ^
  --mode single ^
  --model "%~dp0weights\best.pt" ^
  --output-dir "%~dp0calibration_suite\workspace\shipping_label_batch_test" ^
  %*

endlocal
