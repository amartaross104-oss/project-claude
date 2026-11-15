# One-command update: downloads the latest signalbot code from GitHub and copies it over this folder.
# Your data\ folder (cache, paper_trades.csv, logs) and your scheduled tasks are NOT touched.
# Run from PowerShell:   powershell -ExecutionPolicy Bypass -File scripts\windows\update.ps1
param([string]$Branch = "claude/ai-signal-trade-setup-9g01tf")
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$tmp  = Join-Path $env:TEMP ("signalbot-update-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
  $zip = Join-Path $tmp "src.zip"
  $url = "https://github.com/amartaross104-oss/project-claude/archive/refs/heads/$Branch.zip"
  Write-Host "Memuat turun $Branch ..."
  Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
  Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "x") -Force
  $src = Get-ChildItem (Join-Path $tmp "x") -Directory | Select-Object -First 1
  if (-not $src -or -not (Test-Path (Join-Path $src.FullName "signalbot"))) { throw "Zip dari GitHub tidak seperti dijangka." }
  Get-ChildItem $src.FullName -Force | Where-Object { $_.Name -notin @("data", ".git") } |
    ForEach-Object { Copy-Item $_.FullName -Destination $root -Recurse -Force }
  Write-Host "Kod dikemas kini. Memasang kebergantungan jika perlu ..."
  python -m pip install -r (Join-Path $root "requirements.txt") --quiet --disable-pip-version-check
  Write-Host "Siap. Folder: $root"
} finally {
  Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
}
