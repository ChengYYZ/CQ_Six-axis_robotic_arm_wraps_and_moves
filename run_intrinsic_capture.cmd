@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\orbbec_intrinsic_calibration.py" capture --cols 5 --rows 8 --square-size-mm 25 --save-dir "%~dp0calibration_suite\workspace\intrinsics\images" %*
