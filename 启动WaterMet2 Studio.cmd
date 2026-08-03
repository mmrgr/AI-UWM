@echo off
cd /d "%~dp0"
title WaterMet2 Studio
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo.
echo ==========================================
echo        WaterMet2 Studio Launcher
echo ==========================================
echo.
python -m watermet2_repro.studio
if errorlevel 1 (
  echo.
  echo Startup failed. Review the error above.
  pause
)
