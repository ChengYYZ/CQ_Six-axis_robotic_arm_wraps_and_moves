param(
    [string]$PythonVersion = "3.8",
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptRoot

if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $VenvPath = Join-Path $projectRoot ".venv"
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
$venvPip = Join-Path $VenvPath "Scripts\pip.exe"
$venvSitePackages = Join-Path $VenvPath "Lib\site-packages"
$reusePthPath = Join-Path $venvSitePackages "project0714_orbbec_reuse.pth"

if (-not (Test-Path $VenvPath)) {
    Write-Host "Creating virtual environment at $VenvPath"
    & py "-$PythonVersion" -m venv $VenvPath
}

New-Item -ItemType Directory -Path $venvSitePackages -Force | Out-Null

$reuseCandidates = @(
    "D:\RoboticARMGrasping\TCP-IP\.venv\Lib\site-packages",
    "D:\orbbec_pyenv\Lib\site-packages"
)

$existingReuseCandidates = @()
foreach ($candidate in $reuseCandidates) {
    if (Test-Path $candidate) {
        $existingReuseCandidates += $candidate
    }
}

$reused = $false
if ($existingReuseCandidates.Count -gt 0) {
    Write-Host "Trying local package reuse:"
    foreach ($candidate in $existingReuseCandidates) {
        Write-Host "  $candidate"
    }
    Set-Content -Path $reusePthPath -Value $existingReuseCandidates -Encoding ASCII

    $validation = & $venvPython -c "import importlib.util, cv2; required=('numpy','cv2','pyorbbecsdk'); missing=[m for m in required if importlib.util.find_spec(m) is None]; missing += ([] if hasattr(cv2, 'calibrateHandEye') else ['cv2.calibrateHandEye']); print('OK' if not missing else 'MISSING:' + ','.join(missing))"
    if ($LASTEXITCODE -eq 0 -and $validation -eq "OK") {
        Write-Host "Local reuse succeeded."
        $reused = $true
    }
}

if (-not $reused) {
    Write-Host "Local reuse unavailable; falling back to pip installation."
    Write-Host "Upgrading pip/setuptools/wheel"
    & $venvPython -m pip install --upgrade pip setuptools wheel

    Write-Host "Installing calibration dependencies"
    & $venvPip install -r (Join-Path $scriptRoot "requirements.txt")

    $setupScript = & $venvPython -c "import importlib.util, pathlib; spec = importlib.util.find_spec('pyorbbecsdk'); print((pathlib.Path(spec.origin).resolve().parent / 'scripts' / 'env_setup' / 'setup_env.py') if spec else '')"
    if ($LASTEXITCODE -eq 0 -and $setupScript -and (Test-Path $setupScript)) {
        Write-Host "Running Orbbec environment helper: $setupScript"
        & $venvPython $setupScript
    } else {
        Write-Host "Orbbec environment helper not found; skipping."
    }
}

Write-Host ""
Write-Host "Virtual environment is ready:"
Write-Host "  $VenvPath"
Write-Host ""
Write-Host "Activate with:"
Write-Host "  $VenvPath\Scripts\Activate.ps1"
