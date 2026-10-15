$ErrorActionPreference = 'Stop'

$env:PYTHONUTF8 = '1'
$env:W2A_LEXICON_DIR = 'C:\Users\yeonsu\w2a\L2-KPNS\metadata\lexicons'

Set-Location -LiteralPath $PSScriptRoot

$proc = Start-Process -FilePath 'python' `
    -ArgumentList @(
        '-u', 'server.py',
        '--host', '127.0.0.1',
        '--port', '8000',
        '--quant', 'int8',
        '--dtype', 'bfloat16'
    ) `
    -RedirectStandardOutput 'server.log' `
    -RedirectStandardError 'server.err.log' `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Out-File -FilePath 'server.pid' -Encoding ascii
Write-Host "Server started. PID: $($proc.Id)"
