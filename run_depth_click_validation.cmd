@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\depth_click_validation.py" --width 1280 --height 800 --fps 10 --depth-width 640 --depth-height 400 --depth-fps 5 --wait-timeout-ms 500 %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" (
  echo.
  echo Program exited with code %exitcode%.
  echo Log file: "%~dp0calibration_suite\workspace\validation\depth_click_validation.log"
  pause
)
exit /b %exitcode%
