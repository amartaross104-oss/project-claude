# Makes the existing signalbot scheduled tasks run without a visible window. Safe to run repeatedly;
# it does not ask for your token. Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\hide_windows.ps1
$ErrorActionPreference = "Stop"
$vbs = Join-Path $PSScriptRoot "run_hidden.vbs"
if (-not (Test-Path $vbs)) { throw "run_hidden.vbs not found next to this script" }

foreach ($pair in @(@("SignalbotScalp", "run_scalp.cmd"), @("SignalbotTrendPaper", "run_trend.cmd"))) {
  $name = $pair[0]; $runner = Join-Path $PSScriptRoot $pair[1]
  if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
    $action = New-ScheduledTaskAction -Execute "wscript.exe" -Argument "`"$vbs`" `"$runner`"" `
              -WorkingDirectory $PSScriptRoot
    Set-ScheduledTask -TaskName $name -Action $action | Out-Null
    Write-Host "OK: $name kini berjalan tanpa tetingkap."
  } else {
    Write-Host "Tugas $name tiada, dilangkau."
  }
}
Write-Host "Log masih ditulis ke data\scalp.log dan data\task.log (ralat hanya kelihatan di sana)."
