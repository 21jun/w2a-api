param([string]$Python = 'python', [switch]$SkipModelDownload)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$venvPython = Join-Path $PSScriptRoot '.venv-win\Scripts\python.exe'

& $Python -m venv (Join-Path $PSScriptRoot '.venv-win')
if ($LASTEXITCODE -ne 0) { throw 'Windows virtual environment creation failed.' }
& $venvPython -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
$dependencyFile = Join-Path $PSScriptRoot 'requirements-windows.lock.txt'
if (-not (Test-Path -LiteralPath $dependencyFile)) {
    $dependencyFile = Join-Path $PSScriptRoot 'requirements-windows.txt'
}
& $venvPython -m pip install -r $dependencyFile
if ($LASTEXITCODE -ne 0) { throw 'Project dependency installation failed.' }
& $venvPython (Join-Path $PSScriptRoot 'scripts\patch_windows_bitsandbytes.py')
if ($LASTEXITCODE -ne 0) { throw 'Windows bitsandbytes compatibility patch failed.' }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies are inconsistent.' }
& $venvPython -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda); assert torch.cuda.is_available(), 'CUDA GPU unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda').cpu())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA verification failed.' }
& $venvPython -c "import torch, bitsandbytes as bnb; from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; from qwen_omni_utils import process_mm_info; layer = bnb.nn.Linear8bitLt(64,32,has_fp16_weights=False).cuda(); result = layer(torch.randn(2,64,device='cuda',dtype=torch.float16)); torch.cuda.synchronize(); assert torch.isfinite(result).all(); print('Qwen imports and int8 GPU operation verified.')"
if ($LASTEXITCODE -ne 0) { throw 'Qwen dependency or int8 GPU verification failed.' }
if (-not $SkipModelDownload) {
    if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $PSScriptRoot '.cache\huggingface' }
    $env:HF_HUB_DISABLE_XET = '1'
    & $venvPython -c "import os; from pathlib import Path; from huggingface_hub import snapshot_download; model = os.environ.get('W2A_MODEL', 'Qwen/Qwen2.5-Omni-7B'); print(model if Path(model).exists() else snapshot_download(model, allow_patterns=['*.json','*.safetensors','*.txt','*.tiktoken','*.jinja','*.model','*.pt'], max_workers=4))"
    if ($LASTEXITCODE -ne 0) { throw 'Model download failed; rerun setup to resume the cache.' }
}
Write-Host 'Setup complete. Run .\start-study.cmd to launch the API.'
