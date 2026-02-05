### tools/cts_test.py  ### Version du 05.02.2026 ENRICHI DES BLOCS “QC Normes/Clinique/Mesures”

from __future__ import annotations
import numpy as np

from medgem_poc.qc import iec_bandpass_filter_1d as iec_bandpass_filter

def gen_cal1000(fs: int = 500, dur_s: float = 10.0, amp_mv: float = 1.0) -> np.ndarray:
    n = int(fs * dur_s)
    t = np.arange(n) / fs
    # 1 Hz square: +/- amp/2 around 0
    x = (np.sign(np.sin(2*np.pi*1.0*t)) * (amp_mv/2.0)).astype(np.float32)
    return x

def gen_qrs_triangle(fs: int = 500, dur_s: float = 10.0, amp_mv: float = 1.0) -> np.ndarray:
    n = int(fs * dur_s)
    x = np.zeros(n, dtype=np.float32)
    # put one triangle at 2s
    center = int(2.0 * fs)
    rise_ms = 10
    fall_ms = 10
    rise = int(fs * rise_ms / 1000.0)
    fall = int(fs * fall_ms / 1000.0)
    for i in range(rise):
        x[center - rise + i] = (i + 1) / rise * amp_mv
    for i in range(fall):
        x[center + i] = (1 - (i + 1) / fall) * amp_mv
    return x

def main() -> int:
    fs = 500
    for name, sig in [("CAL1000", gen_cal1000(fs)), ("QRS_TRI", gen_qrs_triangle(fs))]:
        xf, warns = iec_bandpass_filter(sig, fs, notch_hz=50.0)
        # smoke assertions: finite + no explosion + preserve amplitude roughly
        assert np.isfinite(xf).all(), f"{name}: non-finite"
        assert np.std(xf) < 5.0, f"{name}: unstable filter output std={np.std(xf)}"
        # crude amplitude sanity
        assert np.max(np.abs(xf)) > 0.1, f"{name}: amplitude vanished"
        print("OK", name, "warns=", [w["code"] for w in warns])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# Terminus