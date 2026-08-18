@echo off
setlocal
cd /d "%~dp0"

rem Clear broken local proxy placeholders for this process only.
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set GIT_HTTP_PROXY=
set GIT_HTTPS_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set NO_PROXY=*
set no_proxy=*
set YOLO_CONFIG_DIR=%~dp0calibration_suite\workspace\ultralytics

if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

"%~dp0.venv\Scripts\python.exe" "%~dp0yolo_trainning\seg2package2.py" %*
set "exitcode=%errorlevel%"
if not "%exitcode%"=="0" (
  echo.
  echo Program exited with code %exitcode%.
  echo If the error says "No module named clip", install YOLOE CLIP dependency first.
  pause
)
exit /b %exitcode%
