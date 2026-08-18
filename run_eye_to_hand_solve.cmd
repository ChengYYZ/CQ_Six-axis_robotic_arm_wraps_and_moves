@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\orbbec_eye_to_hand_calibration.py" solve --cols 5 --rows 8 --square-size-mm 25 --sample-dir "%~dp0calibration_suite\workspace\eye_to_hand\samples" --intrinsics "%~dp0calibration_suite\workspace\intrinsics\camera_intrinsics.json" --output "%~dp0calibration_suite\workspace\eye_to_hand\eye_to_hand_result.json" %*
