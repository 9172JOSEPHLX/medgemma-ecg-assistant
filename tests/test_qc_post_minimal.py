### tests/test_qc_post_minimal.py ### Feb 12th, 2026 ###
# Feb 2026 — POST QC minimal smoke tests

import numpy as np

from medgem_poc.qc import qc_signal_post_from_leads
from medgem_poc.resample_to_500 import TARGET_FS_HZ


def test_post_fail_on_nonfinite():
    fs = float(TARGET_FS_HZ)
    n = int(10 * fs)

    leads = {
        "II": np.zeros((n,), dtype=np.float32),
        "I": np.ones((n,), dtype=np.float32),
    }
    leads["II"][100] = np.nan  # inject non-finite

    out = qc_signal_post_from_leads(leads, fs_internal_hz=fs)

    assert out["status"] == "FAIL"
    codes = {w.get("code") for w in out.get("warnings", []) if isinstance(w, dict)}
    assert "FAIL_POST_NONFINITE" in codes


def test_post_warn_on_missing_leads_but_pass_if_signal_ok():
    fs = float(TARGET_FS_HZ)
    n = int(10 * fs)

    # Provide only 2 leads, but valid finite + non-constant + moderate amplitude
    t = np.arange(n, dtype=np.float32) / fs
    leads = {
        "I": 0.2 * np.sin(2 * np.pi * 1.0 * t),
        "II": 0.2 * np.sin(2 * np.pi * 1.1 * t),
    }

    out = qc_signal_post_from_leads(leads, fs_internal_hz=fs)

    # Missing leads => WARN (by design)
    assert out["status"] in ("WARN", "PASS")
    codes = {w.get("code") for w in out.get("warnings", []) if isinstance(w, dict)}
    assert "WARN_POST_MISSING_LEADS" in codes

# TERMINUS DU CODE tests/test_qc_post_minimal.py