[CmdletBinding(SupportsShouldProcess = $true)]
param()
$scope = 'API'
$serverPath = Join-Path $PSScriptRoot 'server.py'
$venvPython = Join-Path $PSScriptRoot '.venv-win\Scripts\python.exe'
$cachePaths = @('__pycache__', 'tests\__pycache__', 'scripts\__pycache__', 'wake2adapt\__pycache__', 'wake2adapt\src\__pycache__', '.cache\wheels')

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd([IO.Path]::DirectorySeparatorChar)
$rootPrefix = $projectRoot + [IO.Path]::DirectorySeparatorChar

function Assert-CachePath([string]$relativePath) {
    $target = [IO.Path]::GetFullPath((Join-Path $projectRoot $relativePath))
    if (-not $target.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing cleanup outside this project: $target"
    }
    $current = $projectRoot
    foreach ($part in @('') + ($relativePath -split '[\\/]')) {
        if ($part) { $current = Join-Path $current $part }
        if (Test-Path -LiteralPath $current) {
            $item = Get-Item -LiteralPath $current -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing cleanup through a symlink or junction: $current"
            }
        }
    }
    return $target
}

# Validate every deletion target before stopping services or changing files.
$cacheTargets = @($cachePaths | ForEach-Object { Assert-CachePath $_ })
if ($scope -eq 'API') { $pidPath = Assert-CachePath 'server.pid' }
$snapshot = @(Get-CimInstance Win32_Process -ErrorAction Stop)
$candidates = @($snapshot | Where-Object {
    $process = $_
    if ($scope -eq 'API') {
        $process.Name -in @('python.exe', 'pythonw.exe') -and (
            ($process.CommandLine -and $process.CommandLine.IndexOf($serverPath, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
            ($process.ExecutablePath -eq $venvPython -and $process.CommandLine -match '(?i)\bserver\.py\b')
        )
    } else {
        $process.Name -in @('node.exe', 'workerd.exe') -and (
            ($process.CommandLine -and $process.CommandLine.IndexOf($launcherPath, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
            ($process.CommandLine -and $process.CommandLine.IndexOf($vinextPath, [StringComparison]::OrdinalIgnoreCase) -ge 0) -or
            ($process.ExecutablePath -and $process.ExecutablePath.StartsWith($workerPrefix, [StringComparison]::OrdinalIgnoreCase))
        )
    }
})
$byId = @{}
foreach ($process in $snapshot) { $byId[[int]$process.ProcessId] = $process }
$candidateIds = @{}
foreach ($process in $candidates) { $candidateIds[[int]$process.ProcessId] = $true }

# Stop top-level owners, including every child (GPU workers, logger, workerd).
$roots = @($candidates | Where-Object {
    $ancestorId = [int]$_.ParentProcessId
    $hasOwner = $false
    $visited = @{}
    while ($byId.ContainsKey($ancestorId) -and -not $visited.ContainsKey($ancestorId)) {
        $visited[$ancestorId] = $true
        if ($candidateIds.ContainsKey($ancestorId)) { $hasOwner = $true; break }
        $ancestorId = [int]$byId[$ancestorId].ParentProcessId
    }
    -not $hasOwner
})
$managed = @{}
foreach ($process in $candidates) { $managed[[int]$process.ProcessId] = $process }
do {
    $before = $managed.Count
    foreach ($process in $snapshot) {
        if ($managed.ContainsKey([int]$process.ParentProcessId)) {
            $managed[[int]$process.ProcessId] = $process
        }
    }
} while ($managed.Count -ne $before)

foreach ($process in $roots) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($process.ProcessId)" -ErrorAction Stop
    if (-not $current) { continue }
    if ($current.CreationDate -ne $process.CreationDate) {
        throw "PID $($process.ProcessId) changed; refusing to terminate a reused PID."
    }
    if ($PSCmdlet.ShouldProcess("$($process.Name) PID $($process.ProcessId)", 'Terminate project process tree')) {
        & taskkill.exe /PID "$($process.ProcessId)" /T /F
        if ($LASTEXITCODE -ne 0 -and (Get-Process -Id $process.ProcessId -ErrorAction SilentlyContinue)) {
            throw "Could not stop PID $($process.ProcessId); cache cleanup cancelled."
        }
    }
}
if (-not $WhatIfPreference -and $managed.Count) {
    Start-Sleep -Milliseconds 500
    $remaining = @(Get-CimInstance Win32_Process | Where-Object {
        $managed.ContainsKey([int]$_.ProcessId) -and
        $_.CreationDate -eq $managed[[int]$_.ProcessId].CreationDate
    })
    if ($remaining.Count) {
        throw "Project processes are still running; cache cleanup cancelled: $($remaining.ProcessId -join ', ')"
    }
}
foreach ($target in $cacheTargets) {
    if ((Test-Path -LiteralPath $target) -and $PSCmdlet.ShouldProcess($target, 'Remove generated runtime cache')) {
        Remove-Item -LiteralPath $target -Recurse -Force -ErrorAction Stop
        Write-Host "Removed cache: $target"
    }
}
if ($scope -eq 'API') {
    if ((Test-Path -LiteralPath $pidPath) -and $PSCmdlet.ShouldProcess($pidPath, 'Remove stale PID file')) {
        Remove-Item -LiteralPath $pidPath -Force
    }
}
if ($WhatIfPreference) {
    Write-Host "$scope preview complete. No processes or files were changed."
} else {
    Write-Host "$scope shutdown complete. Local model files and study data are preserved."
}

