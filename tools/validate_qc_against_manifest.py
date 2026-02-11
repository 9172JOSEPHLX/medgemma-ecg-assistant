# tools/validate_qc_against_manifest.py ### Feb 9th, 2026 created
# tools/validate_qc_against_manifest.py ### Feb 11th, 2026 (updated)

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Any, List

import numpy as np

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.fhir_export import qc_to_fhir_observation  # not used but ensures import ok


# --- simple CSV reader (same logic as run_qc_on_csv) ---
def _read_csv_leads(path: str) -> Dict[str, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError(f"CSV has no header: {path}")

        ignore = ("t", "time", "timestamp", "ms", "sec", "s")
        lead_cols = [c for c in r.fieldnames if c and c.strip() and c.strip().lower() not in ignore]
        if not lead_cols:
            raise ValueError(f"No lead columns found in CSV header: {path}")

        buf: Dict[str, List[float]] = {c: [] for c in lead_cols}
        for row in r:
            for c in lead_cols:
                v = row.get(c, "")
                if v is None or str(v).strip() == "":
                    buf[c].append(0.0)
                    continue
                try:
                    buf[c].append(float(v))
                except Exception:
                    buf[c].append(0.0)

    return {k: np.asarray(v, dtype=np.float32) for k, v in buf.items()}


def _infer_notch_hz_from_filename(fname: str, default_hz: float = 50.0) -> float:
    """
    Demo/manifest alignment:
      - Files containing 'notch_invalid' should be evaluated with an invalid notch_hz,
        so the pipeline emits WARN_IEC_NOTCH_INVALID and the dataset-aligned status becomes WARN.
      - Otherwise, use the default 50Hz.
    """
    f = (fname or "").lower()
    if "notch_invalid" in f:
        return 49.0  # intentionally invalid (not 50/60)
    return float(default_hz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", required=True, help="Path to demo_csv directory")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--sex", default="U")
    ap.add_argument("--iec_normalize", type=int, default=1, help="1=IEC filter on, 0=off (default 1)")
    ap.add_argument(
        "--default_notch_hz",
        type=float,
        default=50.0,
        help="Default notch frequency (Hz) when file is not notch_invalid (default 50).",
    )
    args = ap.parse_args()

    manifest_path = os.path.join(args.demo_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest.get("items", [])
    total = 0
    correct = 0

    confusion = {
        "PASS": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "WARN": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "FAIL": {"PASS": 0, "WARN": 0, "FAIL": 0},
    }

    mismatches: List[Dict[str, Any]] = []

    for it in items:
        file = it["file"]
        expected = it["expected_status"]

        path = os.path.join(args.demo_dir, file)
        leads = _read_csv_leads(path)

        notch_hz = _infer_notch_hz_from_filename(file, default_hz=args.default_notch_hz)

        qc_out = qc_signal_from_leads(
            leads,
            fs_hz=float(args.fs),
            patient_sex=str(args.sex),
            iec_normalize=bool(int(args.iec_normalize)),
            notch_hz=float(notch_hz),
        )

        # Manifest compares against dataset-aligned (pedagogical) status
        predicted = qc_out.get("status_dataset_aligned", "UNKNOWN")

        total += 1
        if predicted == expected:
            correct += 1
        else:
            mismatches.append(
                {
                    "file": file,
                    "expected": expected,
                    "predicted": predicted,
                    "notch_hz_used": notch_hz,
                    "status_strict": qc_out.get("status_strict", qc_out.get("status")),
                    "warnings": qc_out.get("warnings", []),
                }
            )

        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

        print(f"{file:35s} expected={expected:5s} predicted={predicted:5s}")

    accuracy = correct / total if total else 0.0

    report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "mismatches": mismatches,
    }

    out_path = os.path.join(args.demo_dir, "qc_validation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- SUMMARY ---")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(json.dumps(confusion, indent=2))
    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()

# Terminus
