# Start-Dev.ps1 — routine dev (py312 + CUDA) ### Feb 14th, 2026 (updated) 20H50

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "L:\Appli_MedGem_PoC"

# --- Activate venv ---
$activate = ".\.venv-kaggle-py312\Scripts\Activate.ps1"
if (!(Test-Path $activate)) { throw "Venv not found: $activate" }
. $activate

# --- Tokens (persistant -> session) ---
if (-not $env:HF_TOKEN) {
  $env:HF_TOKEN = [Environment]::GetEnvironmentVariable("HF_TOKEN","User")
  if (-not $env:HF_TOKEN) { $env:HF_TOKEN = [Environment]::GetEnvironmentVariable("HF_TOKEN","Machine") }
}
if (-not $env:HUGGINGFACE_HUB_TOKEN -and $env:HF_TOKEN) {
  $env:HUGGINGFACE_HUB_TOKEN = $env:HF_TOKEN
}

Write-Host ("HF_TOKEN set: " + [bool]$env:HF_TOKEN)
Write-Host ("HUGGINGFACE_HUB_TOKEN set: " + [bool]$env:HUGGINGFACE_HUB_TOKEN)

# --- Quick env probes ---
python -c "import sys,os; print('py=',sys.version.split()[0]); print('exe=',sys.executable); print('HF_TOKEN set=',bool(os.getenv('HF_TOKEN')))"
python -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.version.cuda); print('avail=',torch.cuda.is_available()); print('gpu=', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

# --- Sanity / demo ---
python .\tools\sanity_llm_cuda.py
python .\tools\demo_qc_to_llm_json.py

# --- Tests ---
python -m pytest -q

# TERMINUS