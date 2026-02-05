from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from medgem_poc import qc


def _iso_now_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _wrap_qc_v1(raw: dict, *, input_type: str, source_path: str, sample_id: str | None, fs_hz: float | None) -> dict:
    # If already qc.v1, keep it
    if isinstance(raw, dict) and raw.get("schema") == "qc.v1":
        return raw

    status = raw.get("status", "WARN") if isinstance(raw, dict) else "WARN"
    warnings = raw.get("warnings", []) if isinstance(raw, dict) else []
    metrics = raw.get("metrics", {}) if isinstance(raw, dict) else {}

    reasons = raw.get("reasons") if isinstance(raw, dict) else None
    if not reasons:
        reasons = [w.get("code", "WARN_UNKNOWN") for w in warnings if isinstance(w, dict) and w.get("code")]
    if not reasons:
        reasons = ["WARN_UNKNOWN"]

    # ensure warnings codes are included in reasons
    reason_set = set(reasons)
    for w in warnings:
        if isinstance(w, dict) and w.get("code") and w["code"] not in reason_set:
            reasons.append(w["code"])
            reason_set.add(w["code"])

    return {
        "schema": "qc.v1",
        "generated_at": _iso_now_z(),
        "input": {"type": input_type, "source_path": source_path, "sample_id": sample_id, "fs_hz": fs_hz},
        "pipeline": {
            "repo": "Appli_MedGem_PoC",
            "commit": None,
            "qc_module": "medgem_poc.qc",
            "qc_version": getattr(qc, "QC_VERSION", "1.0.0"),
        },
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": metrics,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fs", type=float, default=None)
    ap.add_argument("--sex", default="U", choices=["M", "F", "U"])
    ap.add_argument("--sample_id", default=None)
    args = ap.parse_args()

    leads, fs_guess = qc.load_leads_from_csv(args.csv)
    fs = args.fs or fs_guess or 500.0

    if not hasattr(qc, "qc_signal_from_leads"):
        raise RuntimeError("qc_signal_from_leads is missing in medgem_poc.qc (Fi2).")

    raw = qc.qc_signal_from_leads(leads, fs_hz=fs, patient_sex=args.sex)

    if hasattr(qc, "qc_pack_v1"):
        bundle = qc.qc_pack_v1(
            input_type="csv",
            source_path=args.csv,
            sample_id=args.sample_id,
            fs_hz=fs,
            signal_qc=raw,
        )
    else:
        bundle = _wrap_qc_v1(raw, input_type="csv", source_path=args.csv, sample_id=args.sample_id, fs_hz=fs)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)

    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
