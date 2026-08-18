@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\orbbec_eye_to_hand_calibration.py" capture --robot-ip 192.168.2.160 --cols 5 --rows 8 --square-size-mm 25 --save-dir "%~dp0calibration_suite\workspace\eye_to_hand\samples" %*
