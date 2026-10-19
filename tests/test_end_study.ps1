$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$workspacePrefix = [IO.Path]::GetFullPath($repo).TrimEnd('\') + '\'
$fixture = Join-Path $repo ('.cache\shutdown-test-' + [guid]::NewGuid())
$started = @()
function Assert([bool]$condition, [string]$message) {
    if (-not $condition) { throw $message }
}
function Run-End([string]$directory, [switch]$Preview) {
    $arguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $directory 'end-study.ps1'))
    if ($Preview) { $arguments += '-WhatIf' }
    & powershell.exe @arguments
    Assert ($LASTEXITCODE -eq 0) 'Shutdown command failed'
}
try {
    foreach ($scope in @('api', 'frontend')) {
        $directory = Join-Path $fixture $scope
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $source = if ($scope -eq 'api') { $repo } else { Join-Path $repo 'w2a-user-study' }
        Copy-Item -LiteralPath (Join-Path $source 'end-study.ps1') -Destination $directory
        $cache = if ($scope -eq 'api') { '__pycache__' } else { '.next' }
        foreach ($relative in @($cache, '.cache\huggingface', 'logs', '.wrangler\state', 'data')) {
            $folder = Join-Path $directory $relative
            New-Item -ItemType Directory -Path $folder -Force | Out-Null
            Set-Content -LiteralPath (Join-Path $folder 'sentinel') -Value 'keep'
        }
        Set-Content -LiteralPath (Join-Path $directory 'server.pid') -Value '999999'
        Run-End $directory -Preview
        Assert (Test-Path -LiteralPath (Join-Path $directory $cache)) 'Preview deleted a cache'

        # Fake launchers use real interpreters but never load models or bind ports.
        if ($scope -eq 'api') {
            $script = Join-Path $directory 'server.py'
            Set-Content -LiteralPath $script -Value @'
import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
Path(__file__).with_name('child.pid').write_text(str(child.pid))
time.sleep(120)
'@
            $executable = Join-Path $repo '.venv-win\Scripts\python.exe'
        } else {
            New-Item -ItemType Directory -Path (Join-Path $directory 'scripts') | Out-Null
            $script = Join-Path $directory 'scripts\with-study-logger.mjs'
            Set-Content -LiteralPath $script -Value @'
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';
const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)']);
writeFileSync(new URL('../child.pid', import.meta.url), String(child.pid));
setInterval(() => {}, 1000);
'@
            $executable = (Get-Command node).Source
        }
        $process = Start-Process -FilePath $executable -ArgumentList ('"' + $script + '"') -WindowStyle Hidden -PassThru
        $started += $process.Id
        $childFile = Join-Path $directory 'child.pid'
        for ($attempt = 0; $attempt -lt 50 -and -not (Test-Path -LiteralPath $childFile); $attempt++) {
            Start-Sleep -Milliseconds 100
        }
        Assert (Test-Path -LiteralPath $childFile) 'Fixture launcher did not start'
        $childId = [int](Get-Content -LiteralPath $childFile)
        $started += $childId
        Run-End $directory -Preview
        Assert ([bool](Get-Process -Id $childId -ErrorAction SilentlyContinue)) 'Preview killed a child'
        Run-End $directory
        Assert (-not (Get-Process -Id $process.Id -ErrorAction SilentlyContinue)) 'Launcher survived shutdown'
        Assert (-not (Get-Process -Id $childId -ErrorAction SilentlyContinue)) 'Child survived shutdown'
        Assert (-not (Test-Path -LiteralPath (Join-Path $directory $cache))) 'Cache survived cleanup'
        foreach ($relative in @('.cache\huggingface', 'logs', '.wrangler\state', 'data')) {
            Assert (Test-Path -LiteralPath (Join-Path $directory "$relative\sentinel")) "Preserved data deleted: $relative"
        }
        Run-End $directory
        Write-Output "PASS: $scope preview, process tree, cache cleanup, data preservation, repeated shutdown"
    }
} finally {
    foreach ($processId in $started) {
        if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
            & taskkill.exe /PID "$processId" /T /F | Out-Null
        }
    }
    $absoluteFixture = [IO.Path]::GetFullPath($fixture)
    if (-not $absoluteFixture.StartsWith($workspacePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Refusing fixture cleanup outside workspace'
    }
    if (Test-Path -LiteralPath $absoluteFixture) {
        Remove-Item -LiteralPath $absoluteFixture -Recurse -Force
    }
}
