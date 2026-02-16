### tools/cts_test.py  ### Version 06.02.2026 04H30 PM (IEC filter smoke, backward-compatible) “QC Normes/Clinique/Mesures”

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Import strategy: prefer new API name, fallback to older helper name.
# We keep the local alias "iec_bandpass_filter" for the rest of the script.
# ---------------------------------------------------------------------
from medgem_poc.qc import iec_bandpass_filter, _iec_normalize_leads


def gen_cal1000(fs: int = 500, dur_s: float = 10.0, amp_mv: float = 1.0) -> np.ndarray:
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    x = (np.sign(np.sin(2 * np.pi * 1.0 * t)) * (amp_mv / 2.0)).astype(np.float32)
    return x


def gen_qrs_triangle(fs: int = 500, dur_s: float = 10.0, amp_mv: float = 1.0) -> np.ndarray:
    n = int(fs * dur_s)
    x = np.zeros(n, dtype=np.float32)
    center = int(2.0 * fs)
    rise = int(fs * 10 / 1000.0)
    fall = int(fs * 10 / 1000.0)
    rise = max(1, rise)
    fall = max(1, fall)
    for i in range(rise):
        x[center - rise + i] = (i + 1) / rise * amp_mv
    for i in range(fall):
        x[center + i] = (1 - (i + 1) / fall) * amp_mv
    return x


def _assert_sane(name: str, xf: np.ndarray) -> None:
    assert xf.size > 0, f"{name}: empty"
    assert np.isfinite(xf).all(), f"{name}: non-finite"
    assert float(np.std(xf)) < 5.0, f"{name}: unstable std={np.std(xf)}"
    assert float(np.max(np.abs(xf))) > 0.1, f"{name}: amplitude vanished"


def main() -> int:
    fs = 500

    # (A) 1D wrapper smoke (CAL1000 + triangle)
    for name, sig in [("CAL1000", gen_cal1000(fs)), ("QRS_TRI", gen_qrs_triangle(fs))]:
        xf, warns, metrics = iec_bandpass_filter(sig, fs, notch_hz=50.0, mode="diagnostic", return_metrics=True)
        _assert_sane(name, xf)
        print("OK", name, "warns=", [w.get("code") for w in warns], "ok=", metrics.get("ok"))

    # (B) warnings propagation (force invalid notch)
    leads = {"II": gen_qrs_triangle(fs), "I": gen_cal1000(fs)}
    filtered, w, m = _iec_normalize_leads(leads, fs, notch_hz=400.0, mode="diagnostic")  # invalid => must warn

    assert filtered is not None, "IEC normalize: expected filtered dict"
    for k in leads.keys():
        _assert_sane(f"LEAD_{k}", filtered[k])

    assert isinstance(w, list) and len(w) >= 1, "IEC normalize: expected warnings"
    codes = [ww.get("code") for ww in w if isinstance(ww, dict)]
    assert any(c in ("WARN_IEC_FILTER_NOTCH_INVALID", "WARN_FILTER_NOTCH_INVALID", "WARN_FILTER_BAND_CLIPPED") for c in codes), \
        f"IEC normalize: unexpected warning codes={codes}"

    print("OK IEC warnings propagate:", codes[:8], "metrics_ok=", m.get("ok"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus