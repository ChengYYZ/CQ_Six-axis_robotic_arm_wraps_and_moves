@echo off
setlocal
if "%HIKVISION_PASSWORD%"=="" (
  echo HIKVISION_PASSWORD is not set.
  echo In PowerShell, run: $env:HIKVISION_PASSWORD="your camera password"
  exit /b 2
)
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\surface_cluster_grasp.py" --tool-name tool4 --wobj-name wobj0 --tcp-override-xyz-mm -125.370 87.591 279.781 --tcp-override-rpy-deg 178.740 -32.430 -87.920 --rpy-mode align-normal --normal-mode top-plane --tool-contact-axis minus-z --robot-speed-mm-s 400 --standoff-mm 100 --pickup-down-mm 110 --suction-do-port 6 --secondary-suction-do-port 5 --secondary-suction-offset-y-mm -245 --disable-third-suction --disable-fourth-suction --enable-waybill-inspection --cup-volume-check --primary-verified-placement-angles-deg 0 --secondary-verified-placement-angles-deg 0 --placement-tcp-max-reach-mm 1000 --placement-tcp-y-max-mm 1000 --allow-degraded-placement --acknowledge-verified-tcp %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" (
  echo.
  echo Program exited with code %exitcode%.
  pause
)
exit /b %exitcode%
