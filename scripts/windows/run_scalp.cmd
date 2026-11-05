@echo off
rem 15m scalp check (BTCUSDT). Output is appended to data\scalp.log
cd /d "%~dp0..\.."
if not exist data mkdir data
echo ===== %date% %time% >> data\scalp.log
python -m signalbot scalp BTCUSDT --once --max-per-day 6 >> data\scalp.log 2>&1
