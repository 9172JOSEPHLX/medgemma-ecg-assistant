### tests/test_qc_limb_swap_suspect.py ### Mars 11th, 2026  ### P1-D) Tests pytest (skip si pas de CSV demo local)

from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

import pytest

from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads


def _swap_ra_la(leads: Dict[str, object]) -> Dict[str, object]:
    """
    RA <-> LA mapping (signal-only), expected for standard 12-lead ECG:
      I'   = -I
      II'  = III
      III' = II
      aVR' = aVL
      aVL' = aVR
      aVF' = aVF
      V1..V6 unchanged
    """
    # copy arrays to avoid aliasing surprises
    out: Dict[str, object] = {}
    for k, v in leads.items():
        try:
            out[k] = v.copy()  # type: ignore[attr-defined]
        except Exception:
            out[k] = v

    if "I" in leads:
        out["I"] = -leads["I"]  # type: ignore[operator]
    if "II" in leads and "III" in leads:
        out["II"] = leads["III"]
        out["III"] = leads["II"]

    if "aVR" in leads and "aVL" in leads:
        out["aVR"] = leads["aVL"]
        out["aVL"] = leads["aVR"]

    # aVF unchanged (already copied)
    return out


def _find_demo_09487_ok() -> Path | None:
    demo = Path("data/demo_csv")
    if not demo.exists():
        return None
    cands = sorted(demo.glob("*09487*12leads*.csv"))
    return cands[0] if cands else None


def _warn_codes(qc: dict) -> Set[str]:
    ws = qc.get("warnings") or []
    codes: Set[str] = set()
    for w in ws:
        if isinstance(w, dict):
            c = w.get("code")
            if isinstance(c, str) and c:
                codes.add(c)
        elif isinstance(w, str) and w:
            codes.add(w)
    return codes


def test_ok_has_no_limb_swap_suspect_warning() -> None:
    # Skip if demo not present
    p = _find_demo_09487_ok()
    if p is None:
        pytest.skip("demo CSV not present (data/demo_csv/*09487*12leads*.csv)")

    leads, fs = load_leads_from_csv(str(p))
    qc = qc_signal_from_leads(leads, fs_hz=fs, patient_sex="U")

    # Baseline should stay PASS on the OK case
    assert qc.get("status") == "PASS"

    codes = _warn_codes(qc)
    assert "WARN_LIMB_SWAP_SUSPECT" not in codes


def test_ra_la_swap_triggers_limb_swap_suspect_warning() -> None:
    # Skip if demo not present
    p = _find_demo_09487_ok()
    if p is None:
        pytest.skip("demo CSV not present (data/demo_csv/*09487*12leads*.csv)")

    # Skip if feature absent in qc.py (keeps CI robust)
    import medgem_poc.qc as qcmod

    if not hasattr(qcmod, "detect_limb_reversal"):
        pytest.skip("detect_limb_reversal() not present in qc.py")

    leads, fs = load_leads_from_csv(str(p))
    leads_sw = _swap_ra_la(leads)

    qc = qc_signal_from_leads(leads_sw, fs_hz=fs, patient_sex="U")

    codes = _warn_codes(qc)
    assert "WARN_LIMB_SWAP_SUSPECT" in codes

    # With the “acquisition warning only” rule:
    # - status forced to WARN
    # - strict must NOT become FAIL
    assert qc.get("status") == "WARN"
    assert qc.get("status_strict") != "FAIL"

# Terminus