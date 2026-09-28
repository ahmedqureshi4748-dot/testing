@echo off
rem Double-click to forecast XAUUSD on every timeframe (1d down to 1m) with Kronos-small,
rem using live candles from your open, logged-in MetaTrader 5 terminal.
rem Other symbols or options, from a Command Prompt:  run_xauusd.bat GOLD --mt5 --pred-len 24
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1

set "ARGS=%*"
if "%~1"=="" set "ARGS=XAUUSD --mt5"

echo [1/4] Checking for Python 3.12...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.12 not found, installing it with winget...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo Could not find Python 3.12. Install it from https://www.python.org/downloads/ and run this file again.
        goto :fail
    )
)

echo [2/4] Getting the Kronos model code...
if not exist "Kronos\model\kronos.py" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest -UseBasicParsing https://github.com/shiyu-coder/Kronos/archive/refs/heads/master.zip -OutFile kronos.zip; Expand-Archive kronos.zip -DestinationPath . -Force; Remove-Item kronos.zip; if (Test-Path Kronos) { Remove-Item Kronos -Recurse -Force }; Rename-Item Kronos-master Kronos"
    if errorlevel 1 goto :fail
)

echo [3/4] Installing Python packages. The first time takes a few minutes...
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto :fail
)
if not exist ".venv\installed.ok" (
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r Kronos\requirements.txt yfinance MetaTrader5
    if errorlevel 1 goto :fail
    echo ok> ".venv\installed.ok"
)

echo [4/4] Forecasting %ARGS% ...
".venv\Scripts\python.exe" mtf_forecast.py %ARGS%
if errorlevel 1 goto :fail
start "" "%~dp0output"
echo.
echo Done. The table is above, and the chart and CSV are in the output folder.
pause
exit /b 0

:fail
echo.
echo Something went wrong, see the message above.
pause
exit /b 1
