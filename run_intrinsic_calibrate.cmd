@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\orbbec_intrinsic_calibration.py" calibrate --cols 5 --rows 8 --square-size-mm 25 --image-dir "%~dp0calibration_suite\workspace\intrinsics\images" --output "%~dp0calibration_suite\workspace\intrinsics\camera_intrinsics.json" %*
