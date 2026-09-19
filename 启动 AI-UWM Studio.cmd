@echo off
cd /d "%~dp0"
title AI-UWM Studio
set "PYTHONPATH=%CD%\src;%PYTHONPATH%"
echo.
echo ==========================================
echo        AI-UWM Studio Launcher
echo ==========================================
echo.
python -m aiuwm.studio
if errorlevel 1 (
  echo.
  echo Startup failed. Review the error above.
  pause
)
