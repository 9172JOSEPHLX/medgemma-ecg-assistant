### tools/generate_swap_cases.py ### Mars 11th, 2026  ### P1-A) tools/generate_swap_cases.py (CSV → swaps + manifest)


import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical(name: str) -> str:
    n = name.strip()
    low = n.lower()
    if low in {"i", "lead_i"}:
        return "I"
    if low in {"ii", "lead_ii"}:
        return "II"
    if low in {"iii", "lead_iii"}:
        return "III"
    if low in {"avr", "a vr", "a_vr", "av_r", "aVR".lower()}:
        return "aVR"
    if low in {"avl", "a vl", "a_vl", "av_l", "aVL".lower()}:
        return "aVL"
    if low in {"avf", "a vf", "a_vf", "av_f", "aVF".lower()}:
        return "aVF"
    if low.startswith("v") and low[1:].isdigit():
        k = "V" + low[1:]
        if k in LEADS_12:
            return k
    return n  # keep as-is (time_s, extra cols, etc.)


def _read_csv_columns(csv_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        cols: List[List[float]] = [[] for _ in header]
        for row in r:
            if not row or len(row) != len(header):
                continue
            for i, v in enumerate(row):
                try:
                    cols[i].append(float(v))
                except Exception:
                    cols[i].append(float("nan"))

    out: Dict[str, np.ndarray] = {}
    for name, values in zip(header, cols):
        out[name] = np.asarray(values, dtype=np.float64)
    return header, out


def _detect_time_column(header: List[str]) -> Optional[str]:
    for h in header:
        n = h.strip().lower()
        if n in {"t", "time", "time_s", "time_ms", "sec", "secs", "second", "seconds"}:
            return h
    # fallback: wildcard time*/timestamp*
    for h in header:
        n = h.strip().lower()
        if n.startswith("time") or n.startswith("timestamp"):
            return h
    return None


def _estimate_fs(time_vec: np.ndarray) -> Optional[float]:
    if time_vec.size < 3:
        return None
    t = time_vec.astype(np.float64, copy=False)
    d = np.diff(t)
    d = d[d > 0]
    if d.size == 0:
        return None
    dt = float(np.median(d))
    if dt <= 0:
        return None

    # Try s/ms/us scale by plausibility (ECG 20..5000 Hz)
    common = (250.0, 300.0, 360.0, 400.0, 500.0, 1000.0)
    cands: List[Tuple[float, float]] = []
    for denom in (1.0, 1e-3, 1e-6):  # s, ms, us
        dt_s = dt * denom
        fs = 1.0 / dt_s
        if 20.0 <= fs <= 5000.0:
            score = min(abs(fs - c) for c in common)
            cands.append((score, fs))
    if not cands:
        return None
    return float(round(min(cands, key=lambda x: x[0])[1]))


def _require_leads(cols: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # build canonical lead dict from header columns
    lead_map: Dict[str, np.ndarray] = {}
    for k, v in cols.items():
        ck = _canonical(k)
        if ck in LEADS_12:
            lead_map[ck] = v
    missing = [k for k in LEADS_12 if k not in lead_map]
    if missing:
        raise FileNotFoundError(f"Missing required leads in CSV: {missing}")
    return lead_map


def _apply_swap(leads: Dict[str, np.ndarray], swap_type: str) -> Dict[str, np.ndarray]:
    L = {k: leads[k] for k in LEADS_12}

    if swap_type == "RA_LA":
        return {
            **L,
            "I": -L["I"],
            "II": L["III"],
            "III": L["II"],
            "aVR": L["aVL"],
            "aVL": L["aVR"],
            "aVF": L["aVF"],
        }
    if swap_type == "RA_LL":
        return {
            **L,
            "I": -L["III"],
            "II": -L["II"],
            "III": -L["I"],
            "aVR": L["aVF"],
            "aVL": L["aVL"],
            "aVF": L["aVR"],
        }
    if swap_type == "LA_LL":
        return {
            **L,
            "I": L["II"],
            "II": L["I"],
            "III": -L["III"],
            "aVR": L["aVR"],
            "aVL": L["aVF"],
            "aVF": L["aVL"],
        }
    raise ValueError(f"Unknown swap_type: {swap_type}")


def _write_csv(
    out_path: Path,
    header: List[str],
    cols: Dict[str, np.ndarray],
    swapped_leads: Dict[str, np.ndarray],
) -> None:
    # write preserving original header order, replacing lead columns if they match canonical lead names
    canon_by_header = {h: _canonical(h) for h in header}
    n = len(next(iter(cols.values())))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            row = []
            for h in header:
                c = canon_by_header[h]
                if c in LEADS_12:
                    row.append(swapped_leads[c][i])
                else:
                    row.append(cols[h][i])
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Input 12-lead CSV (time + I..V6)")
    ap.add_argument("--out_dir", default="", type=str, help="Output directory (default: alongside input)")
    ap.add_argument("--manifest", default="", type=str, help="Manifest JSON path (default: out_dir/ swap_manifest.json)")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    header, cols = _read_csv_columns(in_path)
    time_col = _detect_time_column(header)
    time_vec = cols[time_col] if time_col else None

    leads = _require_leads(cols)
    fs = _estimate_fs(time_vec) if time_vec is not None else None
    duration_s = None
    if time_vec is not None and time_vec.size >= 2:
        duration_s = float(np.nanmax(time_vec) - np.nanmin(time_vec))

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_path.stem
    outputs = []
    for swap in ("RA_LA", "RA_LL", "LA_LL"):
        swapped = _apply_swap(leads, swap)
        out_csv = out_dir / f"{stem}_{swap}_swap.csv"
        _write_csv(out_csv, header, cols, swapped)
        outputs.append(out_csv)

    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else (out_dir / "swap_manifest.json")

    records = []
    for out_csv in outputs:
        rec = {
            "created_utc": _utc_now_iso(),
            "source_csv": str(in_path),
            "swap_type": out_csv.stem.split("_")[-2] + "_" + out_csv.stem.split("_")[-1].replace("swap", "").strip("_"),
            "output_csv": str(out_csv),
            "sha256": _sha256_file(out_csv),
            "n_samples": int(len(next(iter(cols.values())))),
            "fs_hz": fs,
            "duration_s": duration_s,
        }
        records.append(rec)

    manifest = {
        "schema": "swap_manifest_v1",
        "source_sha256": _sha256_file(in_path),
        "source_csv": str(in_path),
        "fs_hz_estimated": fs,
        "duration_s": duration_s,
        "records": records,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ wrote: {manifest_path}")
    for p in outputs:
        print("✅ wrote:", p.name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus 