# One-time setup. Saves your Telegram credentials as USER environment variables (not in any file in
# this project) and registers a daily Windows scheduled task. No admin rights needed.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File scripts\windows\setup_task.ps1
param(
  [string]$Time = "08:20",               # LOCAL time. The daily candle closes 00:00 UTC = 08:00 Malaysia.
  [string]$TaskName = "SignalbotTrendPaper"
)
$ErrorActionPreference = "Stop"

$runner = Join-Path $PSScriptRoot "run_trend.cmd"
if (-not (Test-Path $runner)) { throw "run_trend.cmd not found next to this script" }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { throw "Python not found in PATH" }

$chat = (Read-Host "Telegram Chat ID (nombor sahaja)").Trim()
$sec  = Read-Host "Tampal token bot (tidak dipaparkan)" -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)
try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr).Trim() }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }

if ($chat -notmatch '^-?\d+$') { throw "Chat ID mesti nombor (contoh 647787296)" }
if ($token -notmatch '^\d+:[A-Za-z0-9_-]{30,}$') { throw "Format token tidak sah. Salin penuh dari BotFather." }

[Environment]::SetEnvironmentVariable("TELEGRAM_BOT_TOKEN", $token, "User")
[Environment]::SetEnvironmentVariable("TELEGRAM_CHAT_ID", $chat, "User")

$vbs      = Join-Path $PSScriptRoot "run_hidden.vbs"   # runs the .cmd with no visible window
$action   = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`" `"$runner`"" `
            -WorkingDirectory (Split-Path $runner -Parent)
$trigger  = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
            -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
  -Description "signalbot: daily PAPER trend check, sends Telegram only on a new breakout" -Force | Out-Null

Write-Host ""
Write-Host "Siap. Tugas '$TaskName' akan jalan setiap hari pukul $Time (waktu komputer ini)."
Write-Host "Uji sekarang:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "Lihat log:      Get-Content data\task.log -Tail 30"
Write-Host "Buang tugas:    Unregister-ScheduledTask -TaskName $TaskName -Confirm:`$false"
