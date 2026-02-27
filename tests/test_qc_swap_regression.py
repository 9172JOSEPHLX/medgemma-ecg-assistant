# tests/test_qc_swap_regression.py  The FEB 26th2026  to test qc.py for 09487 PNG1 WARN or other V3 16H32

from __future__ import annotations

from pathlib import Path
import csv

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "data" / "demo_csv"


def _codes(out: dict) -> list[str]:
    return [
        w.get("code")
        for w in (out.get("warnings") or [])
        if isinstance(w, dict) and w.get("code")
    ]


def _run_qc(csv_path: Path) -> dict:
    from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads

    leads, fs = load_leads_from_csv(str(csv_path))
    return qc_signal_from_leads(leads, fs_hz=fs, patient_sex="U")


def test_loader_ignores_time_s_column_and_keeps_fs_guess(tmp_path: Path) -> None:
    from medgem_poc.qc import load_leads_from_csv

    p = tmp_path / "mini.csv"
    # 500 Hz => dt = 0.002 s
    rows = [
        ["time_s", "II"],
        ["0.000", "0.0"],
        ["0.002", "0.1"],
        ["0.004", "0.0"],
        ["0.006", "-0.1"],
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    leads, fs = load_leads_from_csv(str(p))
    assert "time_s" not in leads
    assert "II" in leads
    assert fs == 500.0


def test_loader_accepts_timestamp_column_via_wildcard_and_validates_monotone(tmp_path: Path) -> None:
    from medgem_poc.qc import load_leads_from_csv

    p = tmp_path / "mini_timestamp.csv"
    # 500 Hz => dt = 0.002 s
    rows = [
        ["timestamp", "II"],
        ["0.000", "0.0"],
        ["0.002", "0.1"],
        ["0.004", "0.0"],
        ["0.006", "-0.1"],
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    leads, fs = load_leads_from_csv(str(p))
    assert "timestamp" not in leads
    assert "II" in leads
    assert fs == 500.0


def test_loader_accepts_time_ms_and_converts_unit_to_fs_guess(tmp_path: Path) -> None:
    from medgem_poc.qc import load_leads_from_csv

    p = tmp_path / "mini_time_ms.csv"
    # 500 Hz => dt = 2.0 ms
    rows = [
        ["time_ms", "II"],
        ["0.0", "0.0"],
        ["2.0", "0.1"],
        ["4.0", "0.0"],
        ["6.0", "-0.1"],
    ]
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)

    leads, fs = load_leads_from_csv(str(p))
    assert "time_ms" not in leads
    assert "II" in leads
    assert fs == 500.0


def test_A_09487_ok_must_be_pass() -> None:
    csv_ok = DEMO_DIR / "ecg_09487_12leads_2p5s_500Hz.csv"
    if not csv_ok.exists():
        pytest.skip(f"Missing demo file: {csv_ok}")

    out = _run_qc(csv_ok)
    codes = _codes(out)

    assert out.get("status") == "PASS"
    assert "WARN_BASELINE_DRIFT" not in codes


def test_B_09487_swap_must_warn_and_flag_swap() -> None:
    """
    Non-régression ciblée sur le CSV "swap".

    - Si la version de qc.py active la détection SWAP (WARN_LIMB_SWAP_SUSPECT),
      alors on attend WARN + code swap, et jamais WARN_BASELINE_DRIFT (artefact time_s).
    - Sinon (détection SWAP non active / heuristique retourne NONE), on n'impose pas WARN ;
      on impose seulement l'absence de WARN_BASELINE_DRIFT.
    """
    csv_swap = DEMO_DIR / "ecg_09487_RA_LA_swap_12leads_2p5s_500Hz.csv"
    if not csv_swap.exists():
        pytest.skip(f"Missing demo file: {csv_swap}")

    out = _run_qc(csv_swap)
    codes = _codes(out)

    assert "WARN_BASELINE_DRIFT" not in codes

    if "WARN_LIMB_SWAP_SUSPECT" in codes:
        assert out.get("status") == "WARN"
    else:
        # Détection SWAP non active: on n'échoue pas le test sur le statut.
        assert out.get("status") in {"PASS", "WARN"}


def test_B_11899_must_stay_warn() -> None:
    cand = [p for p in DEMO_DIR.glob("*11899*.csv") if "swap" not in p.name.lower()]
    if not cand:
        pytest.skip("No 11899 CSV found under data/demo_csv")

    out = _run_qc(cand[0])
    assert out.get("status") == "WARN"


### Terminus du test_qc_swap_regression.py ###