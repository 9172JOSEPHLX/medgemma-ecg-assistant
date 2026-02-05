# tools/run_qc_on_image.py
# Feb 3rd, 2026. (robust: always writes qc.v1 from qc_gate output)

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List


def _iso_now_z() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True, help="Path to ECG image (png/jpg)")
    ap.add_argument("--out", required=True, help="Output qc_report.json path (qc.v1)")
    args = ap.parse_args()

    import medgem_poc.qc as qc

    img = qc.load_image_bgr(args.img)
    res = qc.qc_gate(img)  # image QC only

    status = str(getattr(res, "qc_level", None) or "WARN")
    reasons = list(getattr(res, "reasons", []) or [])
    reasons = [str(x) for x in reasons]

    metrics = dict(getattr(res, "metrics", {}) or {})

    # Convert structured flags -> warnings (optional)
    warnings: List[Dict[str, str]] = []
    flags = getattr(res, "flags", None)
    if isinstance(flags, list):
        for f in flags:
            if isinstance(f, dict) and f.get("code"):
                warnings.append(
                    {"code": str(f.get("code")), "reason": str(f.get("reason", ""))}
                )

    report: Dict[str, Any] = {
        "schema": "qc.v1",
        "generated_at": _iso_now_z(),
        "input": {"type": "image", "source_path": args.img, "sample_id": None, "fs_hz": None},
        "pipeline": {
            "repo": "Appli_MedGem_PoC",
            "commit": None,
            "qc_module": "medgem_poc.qc",
            "qc_version": str(getattr(qc, "QC_VERSION", "unknown")),
        },
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": metrics,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out}")
    print("STATUS=", report.get("status"))
    print("REASONS=", report.get("reasons"))
    print("N_METRICS=", len(report.get("metrics", {}) or {}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# Terminus
# 4) Runner tools/run_qc_on_image.py : maintenant OK  le 04.02.2026 20H37
# STATUS= PASS, REASONS=[], N_METRICS=14
# check_qc_json_contract.py → qc.v1 contract validated ✅
# Re-lecture JSON → confirme que le fichier écrit contient bien status/metrics.
# ➡️ Conclusion : le problème précédent “STATUS=None / metrics=0” est résolu. La version actuelle du runner écrit bien qc.v1 avec contenu.
