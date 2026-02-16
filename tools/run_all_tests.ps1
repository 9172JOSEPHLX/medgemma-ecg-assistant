Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Run from repo root
Set-Location (Split-Path $PSScriptRoot -Parent)

# Activate venv
. .\.venv-ecg-torch310\Scripts\Activate.ps1

# Ensure imports resolve
$env:PYTHONPATH="$PWD\src"

Write-Host "== Compile =="
python -m py_compile .\src\medgem_poc\qc.py
python -m py_compile .\tools\validate_qc_against_manifest.py
python -m py_compile .\tools\run_qc_on_csv.py

Write-Host "== Import sanity =="
python -c "import medgem_poc.qc as m; print('qc.py:', m.__file__)"
python -c "import tools.validate_qc_against_manifest as t; print('validator ok')"

Write-Host "== Manifest regression (must be 27/27) =="
python tools\validate_qc_against_manifest.py --demo_dir ".\data\demo_csv" --fs 500 --sex U --iec_normalize 1

$r = Get-Content .\data\demo_csv\qc_validation_report.json -Raw | ConvertFrom-Json
Write-Host ("correct={0} total={1} accuracy={2}" -f $r.correct, $r.total, $r.accuracy)

if ($r.correct -ne $r.total) {
  throw "QC manifest not fully passing yet"
}

Write-Host "ALL TESTS PASSED"
