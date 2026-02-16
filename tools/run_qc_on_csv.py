# tools/run_qc_on_csv.py    ### Feb 10th, 2026 (updated)

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Any, List, Optional

import numpy as np

from medgem_poc.qc import qc_signal_from_leads
from medgem_poc.fhir_export import qc_to_fhir_observation


def _read_csv_leads(path: str) -> Dict[str, np.ndarray]:
    """
    Lit un CSV avec header contenant des noms de leads (I, II, III, aVR, aVL, aVF, V1..V6).
    Ignore les colonnes 't', 'time', 'timestamp' si présentes.
    Valeurs attendues en mV (float).
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("CSV has no header.")

        ignore = {"t", "time", "timestamp", "sec", "s", "ms"}
        lead_cols = [c for c in r.fieldnames if c and c.strip() and c.strip().lower() not in ignore]

        if not lead_cols:
            raise ValueError("No lead columns found (header).")

        buf: Dict[str, List[float]] = {c: [] for c in lead_cols}

        for row in r:
            for c in lead_cols:
                v = row.get(c, "")
                if v is None or str(v).strip() == "":
                    # missing -> NaN; later we can handle by finite filtering
                    buf[c].append(np.nan)
                else:
                    try:
                        buf[c].append(float(v))
                    except Exception:
                        buf[c].append(np.nan)

    leads: Dict[str, np.ndarray] = {}
    for c, arr in buf.items():
        x = np.asarray(arr, dtype=np.float32)

        # basic NaN handling: if too many NaN -> keep as-is (QC will flag)
        if np.isnan(x).any():
            # replace NaN by 0 (simple PoC); alternative: interpolate
            x = np.nan_to_num(x, nan=0.0).astype(np.float32)

        leads[c] = x

    return leads


def _qc_status_robust(qc_out: Dict[str, Any]) -> Optional[str]:
    """
    Robust status getter: avoids printing None if the QC dict uses a different key.
    Preference order:
      1) status
      2) status_strict
      3) qc_status
      4) status_dataset_aligned
    """
    if not isinstance(qc_out, dict):
        return None
    for k in ("status", "status_strict", "qc_status", "status_dataset_aligned"):
        v = qc_out.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _metric_get(qc_out: Dict[str, Any], key: str) -> Optional[float]:
    """
    Fetch a numeric metric safely from qc_out["metrics"][key] if present.
    """
    try:
        m = qc_out.get("metrics", {}) or {}
        v = m.get(key, None)
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _fhir_component_value_string(fhir: Dict[str, Any], code_text: str) -> Optional[str]:
    """
    Finds Observation.component entry by component.code.text and returns valueString (if any).
    """
    try:
        comps = fhir.get("component", []) or []
        for c in comps:
            code = (c or {}).get("code", {}) or {}
            if (code.get("text") or "") == code_text:
                vs = (c or {}).get("valueString", None)
                if isinstance(vs, str) and vs.strip():
                    return vs.strip()
                return None
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV leads (mV).")
    ap.add_argument("--fs", type=float, default=500.0, help="Sampling frequency (Hz).")
    ap.add_argument("--sex", default="U", help="Patient sex: M/F/U (default U).")
    ap.add_argument("--iec_normalize", type=int, default=1, help="1=IEC filter on, 0=off")
    ap.add_argument("--notch_hz", type=float, default=50.0, help="Notch frequency (Hz), e.g., 50 or 60.")
    ap.add_argument("--out_dir", default="outputs/qc_run", help="Output directory.")
    ap.add_argument("--patient_id", default=None, help="FHIR Patient id (optional).")
    ap.add_argument("--device_id", default=None, help="FHIR Device id (optional).")
    ap.add_argument("--effective_datetime", default=None, help="FHIR effectiveDateTime ISO (optional).")

    # Debug helpers (non-breaking)
    ap.add_argument(
        "--print_fhir_qc",
        type=int,
        default=1,
        help="1=print FHIR QC status components (strict+aligned), 0=off",
    )
    ap.add_argument(
        "--print_metrics",
        type=int,
        default=1,
        help="1=print key metrics (baseline drift, SNR, RMS if present), 0=off",
    )

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    leads = _read_csv_leads(args.csv)

    qc_out = qc_signal_from_leads(
        leads=leads,
        fs_hz=float(args.fs),
        patient_sex=str(args.sex),
        iec_normalize=bool(int(args.iec_normalize)),
        notch_hz=float(args.notch_hz) if args.notch_hz is not None else None,
    )

    qc_path = os.path.join(args.out_dir, "qc.json")
    with open(qc_path, "w", encoding="utf-8") as f:
        json.dump(qc_out, f, ensure_ascii=False, indent=2)

    fhir = qc_to_fhir_observation(
        qc_out,
        patient_id=args.patient_id,
        device_id=args.device_id,
        effective_datetime=args.effective_datetime,
    )

    fhir_path = os.path.join(args.out_dir, "observation.fhir.json")
    with open(fhir_path, "w", encoding="utf-8") as f:
        json.dump(fhir, f, ensure_ascii=False, indent=2)

    # --- Logging (robust + useful for baseline drift work) ---
    status = _qc_status_robust(qc_out)
    warnings_n = len(qc_out.get("warnings", []) or [])

    print(f"[OK] Wrote QC  -> {qc_path}")
    print(f"[OK] Wrote FHIR-> {fhir_path}")
    print(f"[QC] status={status} warnings={warnings_n}")

    if bool(int(args.print_metrics)):
        bw = _metric_get(qc_out, "baseline_drift_mv")
        snr = _metric_get(qc_out, "snr_median_db")
        rms = _metric_get(qc_out, "noise_rms_uv")
        # Print only what exists (avoid noisy logs)
        parts = []
        if bw is not None:
            parts.append(f"baseline_drift_mv={bw:.3f}")
        if snr is not None:
            parts.append(f"snr_median_db={snr:.2f}")
        if rms is not None:
            parts.append(f"noise_rms_uv={rms:.1f}")
        if parts:
            print("[QC metrics] " + " ".join(parts))

    if bool(int(args.print_fhir_qc)):
        strict_vs = _fhir_component_value_string(fhir, "QC status (strict)")
        aligned_vs = _fhir_component_value_string(fhir, "QC status (dataset aligned)")
        print(f"[FHIR] QC strict.valueString={strict_vs!r}")
        print(f"[FHIR] QC aligned.valueString={aligned_vs!r}")
        # Hard assert-like check (but do not crash unless you want it)
        if strict_vs is None or aligned_vs is None:
            print("[FHIR][WARN] Missing QC status components valueString (strict/aligned). Check qc_to_fhir_observation/_add_component.")


if __name__ == "__main__":
    main()

# Terminus

