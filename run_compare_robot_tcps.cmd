@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\compare_robot_tcps.py" %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" pause
exit /b %exitcode%
