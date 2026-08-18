@echo off
setlocal
title Project0714 Calibration Launcher

:menu
cls
echo ==============================
echo   Project0714 Calibration
echo ==============================
echo.
echo 1. Intrinsic: Capture
echo 2. Intrinsic: Calibrate
echo 3. Eye-To-Hand: Capture
echo 4. Eye-To-Hand: Solve
echo 5. Exit
echo.
set /p choice=Please choose [1-5]: 

if "%choice%"=="1" call "%~dp0run_intrinsic_capture.cmd"
if "%choice%"=="2" call "%~dp0run_intrinsic_calibrate.cmd"
if "%choice%"=="3" call "%~dp0run_eye_to_hand_capture.cmd"
if "%choice%"=="4" call "%~dp0run_eye_to_hand_solve.cmd"
if "%choice%"=="5" goto end

echo.
echo Press any key to return to the menu...
pause >nul
goto menu

:end
endlocal

