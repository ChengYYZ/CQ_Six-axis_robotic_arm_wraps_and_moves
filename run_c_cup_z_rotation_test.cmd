@echo off
setlocal
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul
"%~dp0.venv\Scripts\python.exe" "%~dp0calibration_suite\test_c_cup_z_rotation.py" %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" pause
exit /b %exitcode%
