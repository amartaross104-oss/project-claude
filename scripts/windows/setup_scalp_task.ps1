# Registers a Windows scheduled task that runs the 15m scalp check every 15 minutes, 20 seconds after
# each candle closes. Run setup_task.ps1 FIRST (it stores your Telegram token and Chat ID).
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_scalp_task.ps1
param([string]$TaskName = "SignalbotScalp")
$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "run_scalp.cmd"
if (-not (Test-Path $runner)) { throw "run_scalp.cmd not found next to this script" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found in PATH" }
foreach ($v in "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") {
  if (-not [Environment]::GetEnvironmentVariable($v, "User")) {
    throw "$v belum disimpan. Jalankan scripts\windows\setup_task.ps1 dahulu."
  }
}

# First run at the next quarter-hour mark + 20 s, then every 15 minutes.
$now = Get-Date
$mins = [math]::Ceiling(($now - $now.Date).TotalMinutes / 15) * 15
$start = $now.Date.AddMinutes($mins).AddSeconds(20)
if ($start -le $now) { $start = $start.AddMinutes(15) }

$action   = New-ScheduledTaskAction -Execute $runner -WorkingDirectory (Split-Path $runner -Parent)
$trigger  = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 15) `
            -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Description "signalbot: 15m scalp check (UNVALIDATED), sends Telegram on a new signal" -Force | Out-Null

Write-Host ""
Write-Host "Siap. '$TaskName' jalan setiap 15 minit, mula $start."
Write-Host "PENTING: jalankan sekali dahulu supaya data 15m (3 tahun) siap dimuat turun (~2-3 minit):"
Write-Host "    python -m signalbot fetch BTCUSDT --interval 15m"
Write-Host "Uji sekarang:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "Lihat log:      Get-Content data\scalp.log -Tail 30"
Write-Host "Buang tugas:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
