@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0end-study.ps1" %*
exit /b %errorlevel%
