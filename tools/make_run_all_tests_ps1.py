### tools/make_run_all_tests_ps1.py ### Created Feb 11th, 2026

# Generates tools/run_all_tests.ps1 (PowerShell) for QC regression gating.

from __future__ import annotations

from pathlib import Path


PS1_CONTENT = r"""Set-StrictMode -Version Latest
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
"""


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "tools" / "run_all_tests.ps1"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write as UTF-8 (no BOM) to keep git diffs clean.
    out_path.write_text(PS1_CONTENT, encoding="utf-8", newline="\n")
    print(f"[OK] Wrote: {out_path}")


if __name__ == "__main__":
    main()

# Terminus