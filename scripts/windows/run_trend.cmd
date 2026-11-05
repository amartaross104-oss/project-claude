@echo off
rem Daily paper trend check (crypto only). Output is appended to data\task.log
cd /d "%~dp0..\.."
if not exist data mkdir data
echo ===== %date% %time% >> data\task.log
python -m signalbot trend BTCUSDT ETHUSDT --system 2 --paper --notify >> data\task.log 2>&1
