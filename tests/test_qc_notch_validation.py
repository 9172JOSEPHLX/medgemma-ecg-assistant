### tests/test_qc_notch_validation.py ### Feb 12th, 2026 ###

# tests/test_qc_notch_validation.py
# Feb 2026 — Notch validation contract tests

import numpy as np
import math

from medgem_poc.qc import qc_signal_from_leads


def _mk_signal(fs: float, seconds: float = 10.0, hz: float = 1.0):
    n = int(round(seconds * fs))
    t = np.arange(n, dtype=np.float32) / fs
    return (0.2 * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def _mk_leads(fs: float):
    I = _mk_signal(fs, hz=1.0)
    II = _mk_signal(fs, hz=1.1)
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    return {
        "I": I,
        "II": II,
        "III": III,
        "aVR": aVR,
        "aVL": aVL,
        "aVF": aVF,
    }


def test_notch_invalid_emits_warning():
    fs = 500.0
    leads = _mk_leads(fs)

    out = qc_signal_from_leads(
        leads,
        fs_hz=fs,
        patient_sex="U",
        iec_normalize=True,
        notch_hz=55.0,  # invalid
    )

    codes = {w.get("code") for w in out.get("warnings", []) if isinstance(w, dict)}

    assert "WARN_IEC_NOTCH_INVALID" in codes
    assert out["status"] in ("WARN", "FAIL", "PASS")  # no crash, contract respected


def test_notch_valid_50_no_warning():
    fs = 500.0
    leads = _mk_leads(fs)

    out = qc_signal_from_leads(
        leads,
        fs_hz=fs,
        patient_sex="U",
        iec_normalize=True,
        notch_hz=50.0,  # valid
    )

    codes = {w.get("code") for w in out.get("warnings", []) if isinstance(w, dict)}

    assert "WARN_IEC_NOTCH_INVALID" not in codes
    
# Terminus