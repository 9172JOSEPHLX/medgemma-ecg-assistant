# tests/test_qc_limb_swap_three_variants.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads


DEMO_DIR = Path("data/demo_csv")
LIMBS = ["I", "II", "III", "aVR", "aVL", "aVF"]


def _warn_codes(qc: dict) -> set[str]:
    out = set()
    for w in (qc.get("warnings") or []):
        if isinstance(w, dict) and w.get("code"):
            out.add(str(w["code"]))
    return out


def _clone(leads: dict) -> dict:
    return {k: (np.asarray(v).copy() if hasattr(v, "shape") else v) for k, v in leads.items()}


def _swap_ra_la(leads: dict) -> dict:
    L = _clone(leads)
    # RA<->LA: I'=-I; II'=III; III'=II; aVR'=aVL; aVL'=aVR; aVF'=aVF
    if all(k in L for k in LIMBS):
        L["I"] = -L["I"]
        L["II"], L["III"] = L["III"], L["II"]
        L["aVR"], L["aVL"] = L["aVL"], L["aVR"]
        # aVF unchanged
    return L


def _swap_ra_ll(leads: dict) -> dict:
    L = _clone(leads)
    # RA<->LL: I'=-III; II'=-II; III'=-I; aVR'=aVF; aVL'=aVL; aVF'=aVR
    if all(k in L for k in LIMBS):
        I, II, III = L["I"], L["II"], L["III"]
        L["I"] = -III
        L["II"] = -II
        L["III"] = -I
        aVR, aVL, aVF = L["aVR"], L["aVL"], L["aVF"]
        L["aVR"] = aVF
        L["aVL"] = aVL
        L["aVF"] = aVR
    return L


def _swap_la_ll(leads: dict) -> dict:
    L = _clone(leads)
    # LA<->LL: I'=II; II'=I; III'=-III; aVR'=aVR; aVL'=aVF; aVF'=aVL
    if all(k in L for k in LIMBS):
        I, II, III = L["I"], L["II"], L["III"]
        L["I"] = II
        L["II"] = I
        L["III"] = -III
        aVR, aVL, aVF = L["aVR"], L["aVL"], L["aVF"]
        L["aVR"] = aVR
        L["aVL"] = aVF
        L["aVF"] = aVL
    return L


@pytest.mark.parametrize("swap_fn", [_swap_ra_la, _swap_ra_ll, _swap_la_ll])
def test_simulated_swaps_trigger_warn_swap_suspect(swap_fn) -> None:
    p = DEMO_DIR / "ecg_09487_12leads_2p5s_500Hz.csv"
    if not p.exists():
        pytest.skip("demo CSV missing")

    leads, fs = load_leads_from_csv(str(p))
    leads_sw = swap_fn(leads)

    qc = qc_signal_from_leads(leads_sw, fs_hz=fs, patient_sex="U")
    codes = _warn_codes(qc)

    assert "WARN_LIMB_SWAP_SUSPECT" in codes, f"codes={codes}"
    assert qc.get("status") in {"WARN", "FAIL"}  # on veut WARN; FAIL ne doit pas venir du SWAP
    assert qc.get("status_strict") in {"WARN", "FAIL"}
    # important: SWAP ne doit pas transformer strict en FAIL par ce bloc (acquisition warning only)
    # => si FAIL, ce doit être pour d'autres gates (rare sur 09487)

# Terminus