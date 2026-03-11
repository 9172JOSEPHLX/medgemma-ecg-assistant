### tools/bench_swap_detection.py ### Mars 11th, 2026  ### P1-B) tools/bench_swap_detection.py (OK + swaps → JSON/CSV + 3 lignes)

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import csv

from medgem_poc.qc import load_leads_from_csv
import medgem_poc.qc as qcmod


def _guess_truth_from_name(name: str) -> str:
    n = name.lower()
    if "ra_la_swap" in n or "ra_la" in n and "swap" in n:
        return "RA_LA"
    if "ra_ll_swap" in n or "ra_ll" in n and "swap" in n:
        return "RA_LL"
    if "la_ll_swap" in n or "la_ll" in n and "swap" in n:
        return "LA_LL"
    return "OK"


def _normalize_detect_result(x: Any) -> Dict[str, Any]:
    if x is None:
        return {"kind": "NONE", "raw": None}

    if is_dataclass(x):
        d = asdict(x)
        kind = d.get("kind") or d.get("swap") or d.get("label") or "UNKNOWN"
        return {"kind": str(kind), "raw": d}

    if isinstance(x, dict):
        kind = x.get("kind") or x.get("swap") or x.get("label") or x.get("hyp") or "UNKNOWN"
        return {"kind": str(kind), "raw": x}

    if hasattr(x, "kind"):
        try:
            kind = getattr(x, "kind")
            return {"kind": str(kind), "raw": str(x)}
        except Exception:
            return {"kind": "UNKNOWN", "raw": str(x)}

    if isinstance(x, str):
        return {"kind": x, "raw": x}

    return {"kind": "UNKNOWN", "raw": str(x)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str, help="Directory containing OK + *_swap.csv cases")
    ap.add_argument("--out_dir", default="outputs", type=str, help="Output directory")
    args = ap.parse_args()

    if not hasattr(qcmod, "detect_limb_reversal"):
        print("❌ detect_limb_reversal() absent in medgem_poc.qc. Stop.")
        return 2

    in_dir = Path(args.dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = sorted(in_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in: {in_dir}")

    results = []
    ok_total = ok_fp = 0
    swap_total = swap_tp = 0

    for p in csvs:
        leads, fs = load_leads_from_csv(str(p))
        truth = _guess_truth_from_name(p.name)

        try:
            raw = qcmod.detect_limb_reversal(leads)
        except TypeError:
            # some versions may require fs_hz or other params
            try:
                raw = qcmod.detect_limb_reversal(leads, fs_hz=fs)
            except Exception as e:
                raw = {"error": type(e).__name__, "message": str(e)}

        det = _normalize_detect_result(raw)
        det_kind = str(det.get("kind", "NONE")).upper()
        detected = det_kind not in {"NONE", "", "NO", "OK"}

        if truth == "OK":
            ok_total += 1
            if detected:
                ok_fp += 1
        else:
            swap_total += 1
            if detected:
                swap_tp += 1

        results.append(
            {
                "file": str(p),
                "truth": truth,
                "fs_hz": fs,
                "detected": detected,
                "det_kind": det.get("kind"),
                "raw": det.get("raw"),
            }
        )

    # outputs
    json_path = out_dir / "bench_swap_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "bench_swap_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "truth", "fs_hz", "detected", "det_kind"])
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print(f"✅ wrote: {json_path}")
    print(f"✅ wrote: {csv_path}")

    # 3-line summary (no spam)
    print(f"OK cases:   {ok_total} | false positives: {ok_fp}")
    print(f"SWAP cases: {swap_total} | detected: {swap_tp}")
    print("detect_limb_reversal present: yes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus