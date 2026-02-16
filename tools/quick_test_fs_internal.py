### tools\quick_test_fs_internal.py ### Feb 12th, 2026 ###

import math
import numpy as np

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.resample_to_500 import TARGET_FS_HZ, resample_leads_to_500hz


def mk(fs: float, seconds: float = 10.0, hz: float = 1.0, amp_mv: float = 0.2) -> np.ndarray:
    n = int(round(seconds * float(fs)))
    t = np.arange(n, dtype=np.float32) / float(fs)
    return (float(amp_mv) * np.sin(2.0 * math.pi * float(hz) * t)).astype(np.float32)


def mk_leads(fs: float) -> dict:
    I = mk(fs, hz=1.0)
    II = mk(fs, hz=1.1)
    III = II - I
    aVR = -(I + II) / 2.0
    aVL = I - II / 2.0
    aVF = II - I / 2.0
    return {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}


def main() -> int:
    # Quick sanity for the resampler import (optional, but makes failures obvious)
    assert float(TARGET_FS_HZ) == 500.0, f"Unexpected TARGET_FS_HZ={TARGET_FS_HZ}"

    for fs_src in (250.0, 360.0, 500.0):
        leads = mk_leads(fs_src)

        # Optional: verify the helper resampler works on its own
        leads_500, contract, meta = resample_leads_to_500hz(leads, fs_src, prefer_polyphase=True)
        n_src = len(leads["II"])
        n_500 = len(leads_500["II"])

        out = qc_signal_from_leads(leads, fs_hz=fs_src, patient_sex="U", iec_normalize=False, notch_hz=None)
        m = out.get("metrics", {}) or {}
        s = m.get("sampling", {}) or {}

        print("\n--- fs_src =", fs_src, "---")
        print("status:", out.get("status"))
        print("fs_source_hz:", s.get("fs_source_hz"))
        print("fs_internal_hz:", s.get("fs_internal_hz"))
        print("target_fs_hz:", s.get("target_fs_hz"))
        print("resample_applied:", s.get("resample_applied"))
        print("metrics.fs_hz:", m.get("fs_hz"))
        print(f"len(II): src={n_src} -> internal={n_500} (ratio={n_500/max(1,n_src):.3f})")
        if contract is not None:
            print("resample_method:", getattr(contract, "resample_method", None))
            print("resample_ratio:", getattr(contract, "resample_ratio", None))
            print("fs_low_warning:", getattr(contract, "fs_low_warning", None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus