@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\surface_cluster_grasp.py" --dry-run --ignore-workspace-filter %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" (
  echo.
  echo Program exited with code %exitcode%.
  pause
)
exit /b %exitcode%
