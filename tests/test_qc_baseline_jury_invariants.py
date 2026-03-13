# tests/test_qc_baseline_jury_invariants.py

from pathlib import Path

import pytest

from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads


DEMO_DIR = Path("data/demo_csv")


def _warn_codes(qc: dict) -> set[str]:
    out = set()
    for w in (qc.get("warnings") or []):
        if isinstance(w, dict) and w.get("code"):
            out.add(str(w["code"]))
    return out


def test_09487_ok_must_remain_pass_triplet() -> None:
    p = DEMO_DIR / "ecg_09487_12leads_2p5s_500Hz.csv"
    if not p.exists():
        pytest.skip("demo CSV missing")

    leads, fs = load_leads_from_csv(str(p))
    qc = qc_signal_from_leads(leads, fs_hz=fs, patient_sex="U")

    s = qc.get("status")
    st = qc.get("status_strict", s)
    al = qc.get("status_dataset_aligned", st)

    assert (s, st, al) == ("PASS", "PASS", "PASS"), f"got {(s, st, al)} codes={_warn_codes(qc)}"
    assert "WARN_LIMB_SWAP_SUSPECT" not in _warn_codes(qc)

# Terminus

