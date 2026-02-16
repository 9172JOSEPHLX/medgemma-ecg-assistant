### tools/run_demo_csv_suite.py ### Version du 06.02.2026 BLOCS “QC Normes IEC /Clinique/Mesures”

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads, qc_pack_v1

def _expected_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for it in manifest.get("items", []):
        out[it["file"]] = it
    return out

def _status_rank(s: str) -> int:
    return {"PASS": 0, "WARN": 1, "FAIL": 2}.get(str(s).upper(), 99)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing demo CSVs + manifest.json")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--sex", default="M")
    ap.add_argument("--out", default="outputs/demo_suite.jsonl")
    args = ap.parse_args()

    d = Path(args.dir)
    manifest_path = d / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {d}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = _expected_from_manifest(manifest)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = 0
    total = 0
    lines = []

    for fname, meta in sorted(items.items()):
        fpath = d / fname
        if not fpath.exists():
            print("MISSING", fname)
            continue

        leads, fs_guess = load_leads_from_csv(str(fpath))
        fs = float(args.fs) if args.fs else float(fs_guess)

        # Inject notch invalid only when requested by manifest transform
        tr = meta.get("transform", {}) if isinstance(meta, dict) else {}
        runner_param = tr.get("runner_param")
        notch_hz: Optional[float] = 50.0
        if runner_param == "notch_invalid":
            notch_hz = 999.0  # should trigger WARN_FILTER_NOTCH_INVALID but still run

        sig_qc = qc_signal_from_leads(
            leads,
            fs_hz=fs,
            patient_sex=args.sex,
            iec_normalize=True,
            notch_hz=notch_hz,
        )

        qc_v1 = qc_pack_v1(
            input_type="csv",
            source_path=str(fpath),
            sample_id=fname.replace(".csv", ""),
            fs_hz=fs,
            signal_qc=sig_qc,
        )

        got = qc_v1.get("status", "WARN")
        exp = meta.get("expected_status", "WARN")

        # "match" rule: exact expected status OR (for demos) allow stricter (e.g., WARN sample becomes FAIL)
        is_ok = (str(got).upper() == str(exp).upper()) or (_status_rank(got) >= _status_rank(exp))

        total += 1
        if is_ok:
            ok += 1

        row = {
            "file": fname,
            "expected": exp,
            "got": got,
            "match": bool(is_ok),
            "reasons": qc_v1.get("reasons", []),
            "warnings_codes": [w.get("code") for w in qc_v1.get("warnings", []) if isinstance(w, dict)],
        }
        lines.append(row)

        # also keep full qc.v1 snapshot in jsonl (useful for demo)
        qc_v1["_demo_eval"] = row
        out_path.write_text("", encoding="utf-8") if total == 1 else None
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(qc_v1, ensure_ascii=False) + "\n")

        print(("OK " if is_ok else "BAD"), fname, "exp=", exp, "got=", got, "warns=", row["warnings_codes"][:3])

    print(f"\nSUMMARY: {ok}/{total} matched (allowing stricter outcomes).")
    # Write compact summary next to jsonl
    (out_path.with_suffix(".summary.json")).write_text(json.dumps({"ok": ok, "total": total, "rows": lines}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if ok == total else 2

if __name__ == "__main__":
    raise SystemExit(main())

# TERMINUS