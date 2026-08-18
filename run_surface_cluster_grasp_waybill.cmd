@echo off
setlocal
if "%HIKVISION_PASSWORD%"=="" (
  echo HIKVISION_PASSWORD is not set.
  echo In PowerShell, run: $env:HIKVISION_PASSWORD="your camera password"
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\surface_cluster_grasp.py" --tool-name tool4 --wobj-name wobj0 --tcp-override-xyz-mm -125.370 87.591 279.781 --tcp-override-rpy-deg 178.740 -32.430 -87.920 --rpy-mode align-normal --tool-contact-axis minus-z --robot-speed-mm-s 300 --standoff-mm 100 --pickup-down-mm 110 --suction-do-port 5 --secondary-suction-do-port 6 --secondary-suction-offset-y-mm -245 --enable-waybill-inspection %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" (
  echo.
  echo Program exited with code %exitcode%.
  pause
)
exit /b %exitcode%
