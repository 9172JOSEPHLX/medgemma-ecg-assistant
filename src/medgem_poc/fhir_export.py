# src/medgem_poc/fhir_export.py ### Feb 9th, 2026

from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import datetime, timezone


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mk_quantity(value: float, unit: str, ucum_code: str) -> Dict[str, Any]:
    return {
        "value": float(value),
        "unit": unit,
        "system": "http://unitsofmeasure.org",
        "code": ucum_code,
    }


def _add_component(
    components: List[Dict[str, Any]],
    *,
    text: str,
    value: Any,
    unit: Optional[str] = None,
    ucum_code: Optional[str] = None,
) -> None:
    if value is None:
        return

    comp: Dict[str, Any] = {"code": {"text": text}}

    if unit and ucum_code:
        try:
            v = float(value)
        except Exception:
            return
        comp["valueQuantity"] = _mk_quantity(v, unit, ucum_code)
    else:
        comp["valueString"] = str(value)

    components.append(comp)



def qc_to_fhir_observation(
    qc_out: Dict[str, Any],
    *,
    patient_id: Optional[str] = None,
    device_id: Optional[str] = None,
    effective_datetime: Optional[str] = None,
    observation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convertit le dict contract QC:
      {status, reasons[], warnings[{code,reason,(severity)}], metrics{...}}
    en FHIR Observation minimal (interop PoC).
    """

    status_strict = str(qc_out.get("status_strict", qc_out.get("status", "UNKNOWN")))
    status_ds = str(qc_out.get("status_dataset_aligned", "UNKNOWN"))
    reasons = qc_out.get("reasons", []) or []
    warnings = qc_out.get("warnings", []) or []
    metrics = qc_out.get("metrics", {}) or {}

    eff = effective_datetime or _now_iso_z()

    # Minimal FHIR Observation
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [
                {"system": "http://loinc.org", "code": "11502-2", "display": "EKG study"}
            ],
            "text": "12-lead ECG quality control (QC) + derived metrics",
        },
        "effectiveDateTime": eff,
    }

    if observation_id:
        obs["id"] = observation_id

    if patient_id:
        obs["subject"] = {"reference": f"Patient/{patient_id}"}

    if device_id:
        obs["device"] = {"reference": f"Device/{device_id}"}

    # Interpretation (simple, PoC)
    # PASS -> "N" (normal), WARN/FAIL -> "A" (abnormal) for demo triage.
    interp_code = "N" if status_strict == "PASS" else "A"
    obs["interpretation"] = [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                         "code": interp_code}]}]

    # Components (metrics)
    comps: List[Dict[str, Any]] = []

    _add_component(comps, text="QC status (strict)", value=status_strict)
    _add_component(comps, text="QC status (dataset aligned)", value=status_ds)

    _add_component(comps, text="Sampling frequency", value=metrics.get("fs_hz"), unit="Hz", ucum_code="Hz")
    _add_component(comps, text="Heart rate", value=metrics.get("hr_bpm"), unit="beats/minute", ucum_code="/min")

    _add_component(comps, text="PR interval", value=metrics.get("pr_ms_est"), unit="ms", ucum_code="ms")
    _add_component(comps, text="QRS duration", value=metrics.get("qrs_ms_est"), unit="ms", ucum_code="ms")
    _add_component(comps, text="QT interval", value=metrics.get("qt_ms_est"), unit="ms", ucum_code="ms")
    _add_component(comps, text="QTc (Bazett)", value=metrics.get("qtc_ms_est"), unit="ms", ucum_code="ms")

    _add_component(comps, text="SNR (median)", value=metrics.get("snr_med"))  # ratio (no UCUM)
    _add_component(comps, text="Noise RMS", value=metrics.get("noise_rms_uv"), unit="uV", ucum_code="uV")
    _add_component(comps, text="Baseline drift", value=metrics.get("baseline_drift_mv"), unit="mV", ucum_code="mV")

    _add_component(comps, text="Lead used", value=metrics.get("lead_used"))  # string
    _add_component(comps, text="Limb swap hypothesis", value=metrics.get("limb_swap_hypothesis"))  # string

    obs["component"] = comps

    # Notes: warnings + reasons (human-readable)
    note_lines: List[str] = []
    note_lines.insert(0, f"QC strict={status_strict} | benchmark={status_ds}")

    if reasons:
        note_lines.append("Reasons: " + " | ".join([str(r) for r in reasons]))
    if warnings:
        wtxt = []
        for w in warnings:
            if isinstance(w, dict):
                code = w.get("code", "")
                reason = w.get("reason", "")
                sev = w.get("severity", "")
                tag = f"{code}" + (f"[{sev}]" if sev else "")
                wtxt.append(f"{tag}: {reason}".strip())
            else:
                wtxt.append(str(w))
        note_lines.append("Warnings: " + " | ".join(wtxt))

    if note_lines:
        obs["note"] = [{"text": "\n".join(note_lines)}]

    return obs

# Terminus