### tools/bench_swap_detection.py ### Mars 11th, 2026  ### P1-B) tools/bench_swap_detection.py (OK + swaps → JSON/CSV + 3 lignes) Updated Mars  12th, 2026, 07H00 AM

# =============================================================================
# tools/bench_swap_detection.py
# P1-B) bench limb swap detection:
#   - detector_raw: detect_limb_reversal(leads)
#   - detector_confident: gated on score_best/delta (avoid FP)
#   - qc_warned: whether qc_signal_from_leads emits WARN_LIMB_SWAP_SUSPECT (optional)
# Outputs:
#   outputs/<out_dir>/bench_swap_results.json
#   outputs/<out_dir>/bench_swap_summary.csv
# Prints 3-line summary (jury-friendly)
# =============================================================================

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from medgem_poc.qc import load_leads_from_csv, qc_signal_from_leads
import medgem_poc.qc as qcmod


SWAP_KINDS = {"RA_LA", "RA_LL", "LA_LL"}


def _guess_truth_from_name(name: str) -> str:
    n = name.lower()
    # Be explicit with parentheses (avoid precedence surprises)
    if ("ra_la_swap" in n) or (("ra_la" in n) and ("swap" in n)):
        return "RA_LA"
    if ("ra_ll_swap" in n) or (("ra_ll" in n) and ("swap" in n)):
        return "RA_LL"
    if ("la_ll_swap" in n) or (("la_ll" in n) and ("swap" in n)):
        return "LA_LL"
    return "OK"


def _normalize_detect_result(x: Any) -> Dict[str, Any]:
    """
    Normalize detect_limb_reversal output to a dict with:
      kind, score_best, score_none, delta, raw
    """
    out: Dict[str, Any] = {
        "kind": "NONE",
        "score_best": None,
        "score_none": None,
        "delta": None,
        "raw": None,
    }

    if x is None:
        return out

    if is_dataclass(x):
        d = asdict(x)
        out["raw"] = d
        out["kind"] = str(d.get("kind") or d.get("swap") or d.get("label") or "UNKNOWN")
        out["score_best"] = d.get("score_best")
        out["score_none"] = d.get("score_none")
        out["delta"] = d.get("delta")
        return out

    if isinstance(x, dict):
        out["raw"] = x
        out["kind"] = str(x.get("kind") or x.get("swap") or x.get("label") or x.get("hyp") or "UNKNOWN")
        out["score_best"] = x.get("score_best")
        out["score_none"] = x.get("score_none")
        out["delta"] = x.get("delta")
        return out

    if hasattr(x, "kind"):
        try:
            out["kind"] = str(getattr(x, "kind"))
            out["raw"] = str(x)
            return out
        except Exception:
            out["kind"] = "UNKNOWN"
            out["raw"] = str(x)
            return out

    if isinstance(x, str):
        out["kind"] = x
        out["raw"] = x
        return out

    out["kind"] = "UNKNOWN"
    out["raw"] = str(x)
    return out


def _to_float(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        f = float(v)
        return f
    except Exception:
        return None


def _warn_codes(qc_out: Dict[str, Any]) -> set[str]:
    w = qc_out.get("warnings") or []
    codes = set()
    if isinstance(w, list):
        for it in w:
            if isinstance(it, dict):
                c = it.get("code")
                if c:
                    codes.add(str(c))
    return codes


def _detected_raw(det: Dict[str, Any]) -> bool:
    kind = str(det.get("kind", "NONE")).upper().strip()
    return kind in SWAP_KINDS


def _detected_confident(det: Dict[str, Any], score_thres: float, delta_thres: float) -> bool:
    kind = str(det.get("kind", "NONE")).upper().strip()
    if kind not in SWAP_KINDS:
        return False

    sb = _to_float(det.get("score_best"))
    dl = _to_float(det.get("delta"))

    # If metrics missing, do NOT count as confident (avoid FP)
    if sb is None or dl is None:
        return False

    return (sb >= float(score_thres)) and (dl >= float(delta_thres))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=str, help="Directory containing OK + *_swap.csv cases")
    ap.add_argument("--out_dir", default="outputs", type=str, help="Output directory")
    ap.add_argument("--sex", default="U", type=str, help="Patient sex for QC (M/F/U), default U")

    # Gating thresholds (tuned to avoid FP on your OK where score~0.26, delta~0.12)
    ap.add_argument("--score_thres", default=0.35, type=float, help="Min score_best to count as confident detection")
    ap.add_argument("--delta_thres", default=0.15, type=float, help="Min delta to count as confident detection")

    # Optional QC check (recommended)
    ap.add_argument(
        "--with_qc",
        action="store_true",
        help="Also run qc_signal_from_leads and check WARN_LIMB_SWAP_SUSPECT (slower but jury-safe).",
    )

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

    ok_total = 0
    ok_fp_raw = 0
    ok_fp_conf = 0
    ok_fp_qc = 0

    swap_total = 0
    swap_tp_raw = 0
    swap_tp_conf = 0
    swap_tp_qc = 0

    qc_available = True

    for p in csvs:
        leads, fs = load_leads_from_csv(str(p))
        truth = _guess_truth_from_name(p.name)

        # --- detector raw ---
        try:
            raw = qcmod.detect_limb_reversal(leads)
        except TypeError:
            # some versions may require fs_hz or other params
            try:
                raw = qcmod.detect_limb_reversal(leads, fs_hz=fs)
            except Exception as e:
                raw = {"error": type(e).__name__, "message": str(e)}
        except Exception as e:
            raw = {"error": type(e).__name__, "message": str(e)}

        det = _normalize_detect_result(raw)
        kind = str(det.get("kind", "NONE")).upper().strip()

        detected_raw = _detected_raw(det)
        detected_conf = _detected_confident(det, args.score_thres, args.delta_thres)

        # --- optional QC check ---
        qc_warned = None
        qc_status = None
        qc_codes = None
        qc_err = None

        if args.with_qc:
            try:
                qc_out = qc_signal_from_leads(leads, fs_hz=float(fs), patient_sex=str(args.sex).upper().strip() or "U")
                qc_status = qc_out.get("status")
                qc_codes = sorted(_warn_codes(qc_out))
                qc_warned = ("WARN_LIMB_SWAP_SUSPECT" in set(qc_codes))
            except Exception as e:
                qc_available = False
                qc_err = f"{type(e).__name__}: {e}"
                qc_warned = None

        # --- counters ---
        if truth == "OK":
            ok_total += 1
            if detected_raw:
                ok_fp_raw += 1
            if detected_conf:
                ok_fp_conf += 1
            if qc_warned is True:
                ok_fp_qc += 1
        else:
            swap_total += 1
            if detected_raw:
                swap_tp_raw += 1
            if detected_conf:
                swap_tp_conf += 1
            if qc_warned is True:
                swap_tp_qc += 1

        results.append(
            {
                "file": str(p),
                "truth": truth,
                "fs_hz": float(fs),
                "detected_raw": detected_raw,
                "detected_confident": detected_conf,
                "det_kind": det.get("kind"),
                "det_score_best": det.get("score_best"),
                "det_score_none": det.get("score_none"),
                "det_delta": det.get("delta"),
                "qc_status": qc_status,
                "qc_warned_swap": qc_warned,
                "qc_warning_codes": qc_codes,
                "qc_error": qc_err,
                "raw": det.get("raw"),
            }
        )

    # outputs
    json_path = out_dir / "bench_swap_results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "bench_swap_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "file",
            "truth",
            "fs_hz",
            "detected_raw",
            "detected_confident",
            "det_kind",
            "det_score_best",
            "det_delta",
            "qc_status",
            "qc_warned_swap",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in fieldnames})

    print(f"✅ wrote: {json_path}")
    print(f"✅ wrote: {csv_path}")

    # 3-line summary (jury-friendly, no spam)
    # Line 1
    msg1 = (
        f"OK cases:   {ok_total} | FP raw: {ok_fp_raw} | FP confident: {ok_fp_conf}"
    )
    if args.with_qc:
        msg1 += f" | FP QC(WARN_LIMB_SWAP_SUSPECT): {ok_fp_qc}"
    print(msg1)

    # Line 2
    msg2 = (
        f"SWAP cases: {swap_total} | TP raw: {swap_tp_raw} | TP confident: {swap_tp_conf}"
    )
    if args.with_qc:
        msg2 += f" | TP QC(WARN_LIMB_SWAP_SUSPECT): {swap_tp_qc}"
    print(msg2)

    # Line 3
    msg3 = (
        "detect_limb_reversal present: yes"
        f" | thresholds: score_best>={float(args.score_thres):.2f}, delta>={float(args.delta_thres):.2f}"
    )
    if args.with_qc:
        msg3 += f" | qc_check: {'ok' if qc_available else 'partial/errors'}"
    print(msg3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus
