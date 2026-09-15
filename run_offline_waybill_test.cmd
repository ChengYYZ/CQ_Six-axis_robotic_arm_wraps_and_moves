@echo off
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 IMAGE_OR_DIRECTORY [--mode sequence^|single] [other options]
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\offline_waybill_test.py" %*
exit /b %errorlevel%
