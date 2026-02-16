### tools\qc_fi2_smoke_tests.py  ### Feb 9th, 2026. 11H36PM MOD 15H55

import json
import numpy as np
from medgem_poc.qc import qc_signal_from_leads

def _mk_qrs_train(fs: float, n: int, hr_bpm: float = 75.0, amp_mv: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    rr = int(round(fs * 60.0 / hr_bpm))
    qrs_w = int(round(fs * 0.03))
    t_w = int(round(fs * 0.08))
    for i in range(rr, n - rr, rr):
        a = max(1, qrs_w)
        tri = np.concatenate([
            np.linspace(0, amp_mv, a // 2, endpoint=False),
            np.linspace(amp_mv, 0, a - a // 2, endpoint=True),
        ]).astype(np.float32)
        j0 = i
        j1 = min(n, i + tri.size)
        x[j0:j1] += tri[: (j1 - j0)]

        t0 = i + int(round(fs * 0.20))
        if t0 + t_w < n:
            t = (0.25 * amp_mv) * np.sin(np.linspace(0, np.pi, t_w)).astype(np.float32)
            x[t0:t0 + t_w] += t
    return x


def _add_noise(x: np.ndarray, sigma_uv: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma_mv = sigma_uv / 1000.0
    return (x + rng.normal(0.0, sigma_mv, size=x.size).astype(np.float32)).astype(np.float32)


def _add_drift(x: np.ndarray, fs: float, amp_mv: float, freq_hz: float = 0.33) -> np.ndarray:
    t = np.arange(x.size, dtype=np.float32) / np.float32(fs)
    drift = (amp_mv * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)
    return (x + drift).astype(np.float32)


def _mk_leads_from_ii(ii: np.ndarray) -> dict:
    return {
        "II": ii,
        "I": (0.7 * ii).astype(np.float32),
        "III": (0.5 * ii).astype(np.float32),
        "V1": (0.2 * ii).astype(np.float32),
        "V2": (0.4 * ii).astype(np.float32),
        "V5": (0.6 * ii).astype(np.float32),
    }


def _run_case(name: str, leads: dict, fs: float):
    out = qc_signal_from_leads(leads, fs_hz=fs, patient_sex="U")
    codes = [w.get("code", "") for w in out.get("warnings", []) if isinstance(w, dict)]
    print(f"\n=== {name} ===")
    print("status:", out.get("status"))
    print("codes:", ", ".join(codes) if codes else "(none)")
    m = out.get("metrics", {}) or {}
    keys = ["fs_hz", "snr_med", "baseline_drift_mv", "noise_rms_uv", "iec_normalized", "lead_used"]
    mini = {k: m.get(k) for k in keys if k in m}
    print("metrics:", json.dumps(mini, ensure_ascii=False))


def main():
    # T1: Clean
    fs = 500.0
    n = int(fs * 2.5)
    ii = _mk_qrs_train(fs, n, hr_bpm=75.0, amp_mv=1.0)
    ii = _add_noise(ii, sigma_uv=5.0, seed=1)
    _run_case("T1_CLEAN_PASS", _mk_leads_from_ii(ii), fs)

    # T2: Fs too low
    fs2 = 200.0
    n2 = int(fs2 * 2.5)
    ii2 = _mk_qrs_train(fs2, n2, hr_bpm=75.0, amp_mv=1.0)
    ii2 = _add_noise(ii2, sigma_uv=5.0, seed=2)
    _run_case("T2_FS_TOO_LOW_FAIL", _mk_leads_from_ii(ii2), fs2)

    # T3: Constant lead
    fs3 = 500.0
    n3 = int(fs3 * 2.5)
    ii3 = _mk_qrs_train(fs3, n3, hr_bpm=75.0, amp_mv=1.0)
    ii3 = _add_noise(ii3, sigma_uv=5.0, seed=3)
    leads3 = _mk_leads_from_ii(ii3)
    leads3["V1"] = np.zeros_like(leads3["V1"])
    _run_case("T3_CONSTANT_LEAD_FAIL", leads3, fs3)

    # T4: Dropout plateau
    fs4 = 500.0
    n4 = int(fs4 * 2.5)
    ii4 = _mk_qrs_train(fs4, n4, hr_bpm=75.0, amp_mv=1.0)
    ii4 = _add_noise(ii4, sigma_uv=5.0, seed=4)
    i0 = int(fs4 * 1.2)
    L = int(fs4 * 0.40)
    hold = float(ii4[i0])
    ii4[i0:i0 + L] = hold
    _run_case("T4_DROPOUT_PLATEAU_FAIL", _mk_leads_from_ii(ii4), fs4)

    # T5: Drift + noise
    fs5 = 500.0
    n5 = int(fs5 * 2.5)

    ii5 = _mk_qrs_train(fs5, n5, hr_bpm=75.0, amp_mv=1.0)   # <-- ii5 DOIT être défini ici d’abord
    ii5 = _add_drift(ii5, fs5, amp_mv=0.35, freq_hz=0.25)
    ii5 = _add_noise(ii5, sigma_uv=40.0, seed=5)

    t = np.arange(ii5.size, dtype=np.float32) / np.float32(fs5)  # <-- ensuite seulement
#   ii5 += (0.12 * np.sin(2*np.pi*8.0*t)).astype(np.float32)   # <-- composant 8HZ
#   ii5 += (0.08 * np.sin(2*np.pi*8.0*t)).astype(np.float32)   # <-- composant 8HZ
#   ii5 += (0.06 * np.sin(2*np.pi*2.0*t)).astype(np.float32)   # <-- composant 2HZ
    ii5 += (0.12 * np.sin(2*np.pi*8.0*t)).astype(np.float32)   # <-- composant 8HZ
    ii5 += (0.08 * np.sin(2*np.pi*2.0*t)).astype(np.float32)   # <-- composant 2HZ 

    _run_case("T5_DRIFT_NOISE_WARN", _mk_leads_from_ii(ii5), fs5)

if __name__ == "__main__":
    main()


# Terminus