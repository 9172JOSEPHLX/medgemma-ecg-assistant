### tests/test_qc_fs_gate_iec.py ### ### Feb 12th, 2026 ###

# tests/test_qc_fs_gate_iec.py
# Feb 12th, 2026
# IEC Fs gate tests (source sampling rate)

import numpy as np
import math

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.resample_to_500 import TARGET_FS_HZ


def _mk_signal(fs, seconds=5.0, hz=1.0):
    n = int(round(seconds * fs))
    t = np.arange(n, dtype=np.float32) / fs
    return (0.2 * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def _mk_leads(fs):
    I = _mk_signal(fs, hz=1.0)
    II = _mk_signal(fs, hz=1.1)
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    return {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}


# ----------------------------------------------------------
# 1️⃣ FAIL when fs < 250 Hz
# ----------------------------------------------------------

def test_fs_below_250_should_fail():
    fs_src = 200.0  # below IEC minimum
    leads = _mk_leads(fs_src)

    out = qc_signal_from_leads(
        leads,
        fs_hz=fs_src,
        patient_sex="U",
        iec_normalize=False,
        notch_hz=None,
    )

    assert out["status"] == "FAIL"
    assert "FAIL_FS_TOO_LOW" in out.get("reasons", [])


# ----------------------------------------------------------
# 2️⃣ WARN when 250 <= fs < 500 Hz
# ----------------------------------------------------------

def test_fs_between_250_and_500_should_warn():
    fs_src = 360.0  # typical legacy ECG
    leads = _mk_leads(fs_src)

    out = qc_signal_from_leads(
        leads,
        fs_hz=fs_src,
        patient_sex="U",
        iec_normalize=False,
        notch_hz=None,
    )

    # Should not be FAIL
    assert out["status"] in ("WARN", "PASS")

    # But should emit the WARN flag
    codes = {w.get("code") for w in out.get("warnings", []) if isinstance(w, dict)}
    assert "WARN_FS_BELOW_RECOMMENDED" in codes

# Terminus