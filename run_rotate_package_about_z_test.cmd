@echo off
setlocal
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\rotate_package_about_z_test.py" %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" pause
exit /b %exitcode%
