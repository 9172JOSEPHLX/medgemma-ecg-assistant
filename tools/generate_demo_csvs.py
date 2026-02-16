### tools/generate_demo_csvs.py ### Version du 06.02.2026 BLOCS “QC Normes IEC /Clinique/Mesures”

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np

# Reuse your loader to stay 100% consistent with pipeline expectations
from medgem_poc.qc import load_leads_from_csv

LEAD_ORDER_12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def _ensure_2d(leads: Dict[str, np.ndarray]) -> Tuple[np.ndarray, list[str]]:
    """Return matrix [n, L] and lead names."""
    keys = [k for k in LEAD_ORDER_12 if k in leads]
    if not keys:
        # fallback to any deterministic order
        keys = sorted(list(leads.keys()))
    arrs = [np.asarray(leads[k], dtype=np.float32) for k in keys]
    n = min(a.size for a in arrs)
    X = np.stack([a[:n] for a in arrs], axis=1).astype(np.float32)
    return X, keys

def _write_csv(path: Path, X: np.ndarray, keys: list[str], time: Optional[np.ndarray], fs: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    if time is not None:
        cols.append("time")
    cols.extend(keys)
    header = ",".join(cols)

    if time is None:
        out = X
    else:
        out = np.concatenate([time.reshape(-1,1).astype(np.float32), X], axis=1)

    # Write with numpy for speed + stable format
    np.savetxt(str(path), out, delimiter=",", header=header, comments="", fmt="%.6f")

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def _add_white_noise(X: np.ndarray, snr_target: float, seed: int) -> np.ndarray:
    """Approx SNR = std(signal)/std(noise)."""
    g = _rng(seed)
    sig_std = float(np.std(X)) + 1e-6
    noise_std = sig_std / max(1e-3, float(snr_target))
    noise = g.normal(0.0, noise_std, size=X.shape).astype(np.float32)
    return (X + noise).astype(np.float32)

def _add_baseline_wander(X: np.ndarray, fs: float, amp_mv: float, freq_hz: float, seed: int) -> np.ndarray:
    g = _rng(seed)
    n = X.shape[0]
    t = (np.arange(n, dtype=np.float32) / float(fs)).astype(np.float32)
    # random phase per lead
    phase = g.uniform(0, 2*np.pi, size=(1, X.shape[1])).astype(np.float32)
    bw = (amp_mv * np.sin(2*np.pi*freq_hz*t).reshape(-1,1).astype(np.float32))
    bw = bw * np.cos(phase) + (amp_mv * 0.15) * np.sin(2*np.pi*(freq_hz*1.7)*t).reshape(-1,1).astype(np.float32)
    return (X + bw).astype(np.float32)

def _scale(X: np.ndarray, factor: float) -> np.ndarray:
    return (X * float(factor)).astype(np.float32)

def _dropout(X: np.ndarray, fs: float, start_s: float, dur_s: float) -> np.ndarray:
    Y = X.copy()
    a = int(round(start_s * fs))
    b = int(round((start_s + dur_s) * fs))
    a = max(0, min(Y.shape[0], a))
    b = max(0, min(Y.shape[0], b))
    if b > a:
        Y[a:b, :] = 0.0
    return Y.astype(np.float32)

def build_manifest_entry(fname: str, expected: str, transform: Dict[str, Any]) -> Dict[str, Any]:
    return {"file": fname, "expected_status": expected, "transform": transform}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to source 12-lead CSV (2.5s @ 500Hz)")
    ap.add_argument("--out", required=True, help="Output directory (e.g., data/demo_csv)")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    leads, fs_guess = load_leads_from_csv(args.src)
    fs = float(args.fs) if args.fs else float(fs_guess)
    X, keys = _ensure_2d(leads)

    # if time exists in source, rebuild it
    t = (np.arange(X.shape[0], dtype=np.float32) / fs).astype(np.float32)

    manifest: Dict[str, Any] = {
        "schema": "demo.manifest.v1",
        "source_csv": os.path.abspath(args.src),
        "fs_hz": fs,
        "n_samples": int(X.shape[0]),
        "lead_names": keys,
        "items": [],
    }

    base_seed = int(args.seed)

    # ---------------- PASS (9) ----------------
    pass_specs = [
        ("PASS_01_clean.csv", X, {"type": "copy"}),
        ("PASS_02_scale_0p97.csv", _scale(X, 0.97), {"type": "scale", "factor": 0.97}),
        ("PASS_03_scale_1p03.csv", _scale(X, 1.03), {"type": "scale", "factor": 1.03}),
        ("PASS_04_scale_0p99.csv", _scale(X, 0.99), {"type": "scale", "factor": 0.99}),
        ("PASS_05_noise_snr20.csv", _add_white_noise(X, 20.0, base_seed+5), {"type": "noise", "snr": 20.0}),
        ("PASS_06_noise_snr18.csv", _add_white_noise(X, 18.0, base_seed+6), {"type": "noise", "snr": 18.0}),
        ("PASS_07_noise_snr16.csv", _add_white_noise(X, 16.0, base_seed+7), {"type": "noise", "snr": 16.0}),
        ("PASS_08_bw_0p10mv.csv", _add_baseline_wander(X, fs, 0.10, 0.33, base_seed+8), {"type": "baseline_wander", "amp_mv": 0.10, "freq_hz": 0.33}),
        ("PASS_09_bw_0p15mv.csv", _add_baseline_wander(X, fs, 0.15, 0.25, base_seed+9), {"type": "baseline_wander", "amp_mv": 0.15, "freq_hz": 0.25}),
    ]
    for fname, sig, tr in pass_specs:
        _write_csv(out_dir / fname, sig, keys, t, fs)
        manifest["items"].append(build_manifest_entry(fname, "PASS", tr))

    # ---------------- WARN (9) ----------------
    warn_specs = [
        ("WARN_01_bw_0p70mv.csv", _add_baseline_wander(X, fs, 0.70, 0.33, base_seed+11), {"type": "baseline_wander", "amp_mv": 0.70, "freq_hz": 0.33}),
        ("WARN_02_bw_0p85mv.csv", _add_baseline_wander(X, fs, 0.85, 0.27, base_seed+12), {"type": "baseline_wander", "amp_mv": 0.85, "freq_hz": 0.27}),
        ("WARN_03_bw_1p00mv.csv", _add_baseline_wander(X, fs, 1.00, 0.20, base_seed+13), {"type": "baseline_wander", "amp_mv": 1.00, "freq_hz": 0.20}),
        ("WARN_04_bw_0p60mv.csv", _add_baseline_wander(X, fs, 0.60, 0.40, base_seed+14), {"type": "baseline_wander", "amp_mv": 0.60, "freq_hz": 0.40}),
        ("WARN_05_noise_snr9.csv", _add_white_noise(X, 9.0, base_seed+15), {"type": "noise", "snr": 9.0}),
        ("WARN_06_noise_snr8.csv", _add_white_noise(X, 8.0, base_seed+16), {"type": "noise", "snr": 8.0}),
        ("WARN_07_noise_snr7.csv", _add_white_noise(X, 7.0, base_seed+17), {"type": "noise", "snr": 7.0}),
        # The “notch invalid” warnings are injected in the runner (not in CSV)
        ("WARN_08_clean_notch_invalid.csv", X, {"type": "copy", "runner_param": "notch_invalid"}),
        ("WARN_09_bw_0p70mv_notch_invalid.csv", _add_baseline_wander(X, fs, 0.70, 0.30, base_seed+19), {"type": "baseline_wander", "amp_mv": 0.70, "freq_hz": 0.30, "runner_param": "notch_invalid"}),
    ]
    for fname, sig, tr in warn_specs:
        _write_csv(out_dir / fname, sig, keys, t, fs)
        manifest["items"].append(build_manifest_entry(fname, "WARN", tr))

    # ---------------- FAIL (9) ----------------
    fail_specs = [
        ("FAIL_01_noise_snr4.csv", _add_white_noise(X, 4.0, base_seed+21), {"type": "noise", "snr": 4.0}),
        ("FAIL_02_noise_snr3.csv", _add_white_noise(X, 3.0, base_seed+22), {"type": "noise", "snr": 3.0}),
        ("FAIL_03_noise_snr2.csv", _add_white_noise(X, 2.0, base_seed+23), {"type": "noise", "snr": 2.0}),
        ("FAIL_04_noise_snr1p5.csv", _add_white_noise(X, 1.5, base_seed+24), {"type": "noise", "snr": 1.5}),
        ("FAIL_05_noise_snr1.csv", _add_white_noise(X, 1.0, base_seed+25), {"type": "noise", "snr": 1.0}),
        ("FAIL_06_dropout_0p5s.csv", _dropout(X, fs, start_s=0.8, dur_s=0.5), {"type": "dropout", "start_s": 0.8, "dur_s": 0.5}),
        ("FAIL_07_dropout_0p8s.csv", _dropout(X, fs, start_s=1.1, dur_s=0.8), {"type": "dropout", "start_s": 1.1, "dur_s": 0.8}),
        ("FAIL_08_dropout_1p0s.csv", _dropout(X, fs, start_s=0.5, dur_s=1.0), {"type": "dropout", "start_s": 0.5, "dur_s": 1.0}),
        ("FAIL_09_flatline.csv", np.zeros_like(X, dtype=np.float32), {"type": "flatline"}),
    ]
    for fname, sig, tr in fail_specs:
        _write_csv(out_dir / fname, sig, keys, t, fs)
        manifest["items"].append(build_manifest_entry(fname, "FAIL", tr))

    # Write manifest
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OK: wrote {len(manifest['items'])} files + manifest.json to {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# TERMINUS