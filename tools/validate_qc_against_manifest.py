# tools/validate_qc_against_manifest.py ### Feb 9th, 2026 created
# tools/validate_qc_against_manifest.py ### Feb 11th, 2026 (updated)
# tools/validate_qc_against_manifest.py ### Feb 13th, 2026 (updated) 14H00

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from medgem_poc.qc import qc_signal_from_leads


def _infer_notch_hz_from_filename(filename: str, default_hz: float) -> Optional[float]:
    """
    Heuristique dataset: certains fichiers contiennent 'notch_invalid' pour simuler un réglage notch incohérent
    afin que la pipeline émette WARN_IEC_NOTCH_INVALID et que le status dataset-aligned devienne WARN.

    Règles:
      - Par défaut : return default_hz (ex: 50 Hz)
      - Si 'notch_invalid' dans le nom : return 123.0 (volontairement invalide)
        => ton QC déclenche WARN_IEC_NOTCH_INVALID car notch_hz ∉ {50,60}
    """
    name = filename.lower()
    if "notch_invalid" in name:
        return 123.0  # volontairement invalide -> déclenche WARN_IEC_NOTCH_INVALID
    return float(default_hz)


def _load_leads_from_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Lit un CSV (time + 12 leads) et retourne {lead_name: np.ndarray(float32)}.
    Ignore la colonne "time" (case-insensitive).
    """
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        lead_cols = [c for c in cols if c.lower() != "time"]
        buf: Dict[str, List[float]] = {c: [] for c in lead_cols}

        for row in reader:
            for c in lead_cols:
                buf[c].append(float(row[c]))

    return {c: np.asarray(v, dtype=np.float32) for c, v in buf.items()}


def _load_manifest(demo_dir: Path) -> List[Dict[str, Any]]:
    """
    Cherche un manifest dans demo_dir. On supporte plusieurs noms usuels.
    Format attendu: liste de dicts contenant au minimum:
      - file (ou filename) : nom du CSV
      - expected_status (ou expected) : PASS/WARN/FAIL
    """
    candidates = [
        demo_dir / "qc_manifest.json",
        demo_dir / "manifest.json",
        demo_dir / "qc_manifest.csv",
    ]

    for p in candidates:
        if p.exists() and p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "items" in data:
                return list(data["items"])
            if isinstance(data, list):
                return data
            raise ValueError(f"Unexpected JSON manifest shape in {p}")

        if p.exists() and p.suffix.lower() == ".csv":
            items: List[Dict[str, Any]] = []
            with p.open(newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    items.append(dict(row))
            return items

    raise FileNotFoundError(
        f"No manifest found in {demo_dir}. Tried: " + ", ".join(str(x) for x in candidates)
    )


def _norm_expected(item: Dict[str, Any]) -> str:
    for k in ("expected_status", "expected", "status", "label"):
        if k in item and item[k] is not None:
            return str(item[k]).strip().upper()
    raise KeyError(f"Manifest item missing expected status keys: {item}")


def _norm_file(item: Dict[str, Any]) -> str:
    for k in ("file", "filename", "path", "csv"):
        if k in item and item[k] is not None:
            return str(item[k]).strip()
    raise KeyError(f"Manifest item missing filename keys: {item}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo_dir", required=True, help="Directory containing demo CSVs + manifest")
    ap.add_argument("--fs", type=float, default=500.0)
    ap.add_argument("--sex", type=str, default="U")
    ap.add_argument("--iec_normalize", type=int, default=1, help="1=on, 0=off")
    ap.add_argument("--default_notch_hz", type=float, default=50.0)
    args = ap.parse_args()

    demo_dir = Path(args.demo_dir)
    manifest = _load_manifest(demo_dir)

    confusion: Dict[str, Dict[str, int]] = {
        "PASS": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "WARN": {"PASS": 0, "WARN": 0, "FAIL": 0},
        "FAIL": {"PASS": 0, "WARN": 0, "FAIL": 0},
    }

    total = 0
    correct = 0
    mismatches: List[Dict[str, Any]] = []
    per_file: List[Dict[str, Any]] = []

    for it in manifest:
        file = _norm_file(it)
        expected = _norm_expected(it)

        csv_path = demo_dir / file
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        notch_hz = _infer_notch_hz_from_filename(file, default_hz=args.default_notch_hz)

        leads = _load_leads_from_csv(csv_path)

        qc_out = qc_signal_from_leads(
            leads,
            fs_hz=float(args.fs),
            patient_sex=str(args.sex),
            iec_normalize=bool(int(args.iec_normalize)),
            notch_hz=notch_hz,
        )

        # Manifest compares against dataset-aligned (pedagogical) status
        predicted = qc_out.get("status_dataset_aligned", "UNKNOWN")

        # Transparence produit: statut strict (gate)
        status_strict = qc_out.get("status_strict", qc_out.get("status", "UNKNOWN"))

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
                    "status_strict": status_strict,
                    "warnings": qc_out.get("warnings", []),
                }
            )

        if expected in confusion and predicted in confusion[expected]:
            confusion[expected][predicted] += 1

        print(
            f"{file:35s} expected={expected:5s} "
            f"predicted={predicted:5s} strict={status_strict:5s}"
        )

        per_file.append(
            {
                "file": file,
                "expected": expected,
                "predicted_dataset_aligned": predicted,
                "status_strict": status_strict,
                "notch_hz_used": notch_hz,
                "warnings": qc_out.get("warnings", []),
                "reasons": qc_out.get("reasons", []),
                "metrics": qc_out.get("metrics", {}),
            }
        )

    accuracy = correct / total if total else 0.0

    report = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "confusion_matrix": confusion,
        "mismatches": mismatches,
        "per_file": per_file,
    }

    out_path = demo_dir / "qc_validation_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n--- SUMMARY ---")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion matrix:")
    print(json.dumps(confusion, indent=2))
    print(f"\nReport written to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Terminus

