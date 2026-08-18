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
set PIP_NO_INDEX=
set NO_PROXY=*
set no_proxy=*

"%~dp0.venv\Scripts\python.exe" "%~dp0tools\install_yoloe_clip.py"
if errorlevel 1 goto failed

echo.
echo YOLOE CLIP dependency installed successfully.
pause
exit /b 0

:failed
echo.
echo Failed to install YOLOE CLIP dependency.
echo Check that Git is installed and GitHub/PyPI are reachable from this computer.
pause
exit /b 1
