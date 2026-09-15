@echo off
setlocal
cd /d "%~dp0"
"%~dp0.venv\Scripts\pythonw.exe" "%~dp0calibration_suite\waybill_desktop.py"
endlocal
