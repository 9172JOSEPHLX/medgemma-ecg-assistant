### tests/test_qc_sampling_contract.py """ ### Feb 12th, 2026 ###

# tests/test_qc_sampling_contract.py
# Feb 12th, 2026 — Sampling Contract Tests (fs_internal)

import numpy as np
import math

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.resample_to_500 import TARGET_FS_HZ


# ----------------------------
# Synthetic signal generator
# ----------------------------

def _mk(fs, seconds=10.0, hz=1.0):
    n = int(round(seconds * fs))
    t = np.arange(n, dtype=np.float32) / fs
    return (0.2 * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def _mk_leads(fs, seconds=10.0):
    I = _mk(fs, seconds, hz=1.0)
    II = _mk(fs, seconds, hz=1.1)
    III = II - I
    aVR = -(I + II) / 2
    aVL = I - II / 2
    aVF = II - I / 2
    return {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}


# ----------------------------
# Tests
# ----------------------------

def test_fs_internal_is_always_target():
    for fs_src in (250.0, 360.0, 500.0):
        out = qc_signal_from_leads(
            _mk_leads(fs_src),
            fs_hz=fs_src,
            patient_sex="U",
            iec_normalize=False,
            notch_hz=None,
        )

        m = out["metrics"]
        s = m.get("sampling", {})

        assert s["fs_internal_hz"] == TARGET_FS_HZ
        assert m["fs_hz"] == TARGET_FS_HZ


def test_resample_applied_flag():
    # 250 Hz → must resample
    out_250 = qc_signal_from_leads(_mk_leads(250.0), fs_hz=250.0,
                                   patient_sex="U", iec_normalize=False, notch_hz=None)
    assert out_250["metrics"]["sampling"]["resample_applied"] is True

    # 500 Hz → no resample
    out_500 = qc_signal_from_leads(_mk_leads(500.0), fs_hz=500.0,
                                   patient_sex="U", iec_normalize=False, notch_hz=None)
    assert out_500["metrics"]["sampling"]["resample_applied"] is False


def test_internal_length_after_resample():
    fs_src = 250.0
    seconds = 10.0

    leads = _mk_leads(fs_src, seconds)
    out = qc_signal_from_leads(leads, fs_hz=fs_src,
                               patient_sex="U", iec_normalize=False, notch_hz=None)

    # internal resampled signal length check
    resampled, _, _ = qc_signal_from_leads.__globals__["resample_leads_to_500hz"](leads, fs_src)

    for k in resampled:
        assert len(resampled[k]) == int(seconds * TARGET_FS_HZ)
        

# Terminus