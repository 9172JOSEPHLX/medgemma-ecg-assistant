#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def check_and_optional_dedup(qc_out: dict, *, dedup: bool = False) -> dict:
    warnings = qc_out.get("warnings") or []
    reasons  = qc_out.get("reasons") or []

    # normalize warnings -> code
    codes = []
    for w in warnings:
        if isinstance(w, dict):
            codes.append(str(w.get("code") or "").strip() or "(missing_code)")
        else:
            codes.append(str(w).strip() or "(missing_code)")

    by_code = {}
    for i, c in enumerate(codes):
        by_code.setdefault(c, []).append(i)

    dup_codes = {c: idxs for c, idxs in by_code.items() if len(idxs) > 1}

    # reasons duplicates (exact string)
    rmap = {}
    for i, r in enumerate(reasons):
        rr = str(r).strip()
        if rr:
            rmap.setdefault(rr, []).append(i)
    dup_reasons = {r: idxs for r, idxs in rmap.items() if len(idxs) > 1}

    if dedup:
        # keep first per code
        seen = set()
        new_warns = []
        for w in warnings:
            c = (str(w.get("code") or "").strip() if isinstance(w, dict) else str(w).strip()) or "(missing_code)"
            if c in seen:
                continue
            seen.add(c)
            new_warns.append(w)
        qc_out["warnings"] = new_warns

        # keep first per reason string
        seen_r = set()
        new_reasons = []
        for r in reasons:
            rr = str(r).strip()
            if not rr or rr in seen_r:
                continue
            seen_r.add(rr)
            new_reasons.append(r)
        qc_out["reasons"] = new_reasons

    return {
        "warnings_total": len(warnings),
        "warnings_unique_codes": len(by_code),
        "warnings_dup_codes": dup_codes,
        "reasons_total": len(reasons),
        "reasons_dup": dup_reasons,
        "dedup_applied": bool(dedup),
        "warnings_after_dedup": len(qc_out.get("warnings") or []) if dedup else None,
        "reasons_after_dedup": len(qc_out.get("reasons") or []) if dedup else None,
    }

def main() -> int:
    ap = argparse.ArgumentParser(description="Check duplicated warning codes/reasons in qc_out JSON.")
    ap.add_argument("path", help="Path to a qc_out JSON file (must contain warnings/reasons).")
    ap.add_argument("--dedup", action="store_true", help="Write back deduplicated JSON in-place.")
    args = ap.parse_args()

    p = Path(args.path)
    qc_out = json.loads(p.read_text(encoding="utf-8"))

    rep = check_and_optional_dedup(qc_out, dedup=args.dedup)
    print(json.dumps(rep, indent=2))

    if args.dedup:
        p.write_text(json.dumps(qc_out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("✅ wrote back:", str(p))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())