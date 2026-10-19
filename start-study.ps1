param([int]$Port = 8000, [ValidateSet('int8', 'nf4', 'none')][string]$Quant = 'int8')
$ErrorActionPreference = 'Stop'

$env:PYTHONUTF8 = '1'
if (-not $env:W2A_LEXICON_DIR) {
    $lexiconDirectory = Join-Path $PSScriptRoot '..\L2-KPNS\metadata\lexicons'
    if (Test-Path -LiteralPath $lexiconDirectory) { $env:W2A_LEXICON_DIR = $lexiconDirectory }
}
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface' }

Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot '.venv-win\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    throw 'Project Python is missing. Run .\setup-windows.ps1 first.'
}
$pidFile = Join-Path $PSScriptRoot 'server.pid'
if (Test-Path -LiteralPath $pidFile) {
    $existingPidText = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($existingPidText -match '^\d+$') {
        $existingServer = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPidText" -ErrorAction SilentlyContinue
        if ($existingServer -and $existingServer.ExecutablePath -eq $venvPython -and
            $existingServer.CommandLine -match '(?i)\bserver\.py\b') {
            throw "Project server is already running or loading (PID $existingPidText)."
        }
    }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the existing server or choose -Port."
}
$proc = Start-Process -FilePath $venvPython `
    -ArgumentList @(
        '-u', ('"' + (Join-Path $PSScriptRoot 'server.py') + '"'),
        '--host', '127.0.0.1',
        '--port', "$Port",
        '--quant', $Quant,
        '--dtype', 'bfloat16',
        '--retrieval_timeout_s', '20'
    ) `
    -RedirectStandardOutput 'server.log' `
    -RedirectStandardError 'server.err.log' `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File -FilePath 'server.pid' -Encoding ascii
Write-Host "Server started. PID: $($proc.Id)"
Write-Host 'DB search timeout: 20 seconds per attempt. ASR timeout: 10 seconds.'
