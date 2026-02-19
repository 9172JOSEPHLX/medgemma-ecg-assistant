# src/medgem_poc/fhir_export.py ### Feb 9th, 2026

# src/medgem_poc/fhir_export.py  ### Updated Feb 16th, 2026 updated 17h16.  (PATCHED Feb 17th, 2026, 17H32)
#                                ### Updated Feb 19th, 2026 updated 16h45.  (COMMITED Feb 19th, 2026)

# FHIR export (PoC) — QC -> Observation
# - Robust FHIR status mapping (FAIL -> entered-in-error)
# - Category + issued timestamps
# - Interpretation mapping (PASS=N, WARN=A, FAIL=None)
# - Uniform component codes with stable coding system (QC_SYSTEM_URI)
# - UCUM quantities where applicable (including unitless ratios with UCUM code "1")
# - Product-safe behavior: strict FAIL blocks derived metrics export

from __future__ import annotations

from typing import Any, Dict, Optional, List
from datetime import datetime, timezone

UCUM_SYSTEM = "http://unitsofmeasure.org"

### Stable namespace for your internal QC/derived-metrics codes.
### TODO: replace "example.org" with your real domain/namespace when available.
### QC_SYSTEM_URI = "http://example.org/medgemma/qc"

QC_SYSTEM_URI = "https://medgemma.ai/fhir/qc"

# Canonical component codes (stable identifiers)
C_QC_STATUS_STRICT = "QC_STATUS_STRICT"
C_QC_STATUS_DATASET = "QC_STATUS_DATASET_ALIGNED"

C_FS_HZ = "FS_HZ"
C_HR_BPM = "HR_BPM"
C_PR_MS = "PR_MS_EST"
C_QRS_MS = "QRS_MS_EST"
C_QT_MS = "QT_MS_EST"
C_QTC_MS = "QTC_BAZETT_MS_EST"

C_SNR_MED = "SNR_MED"  # unitless ratio
C_NOISE_RMS_UV = "NOISE_RMS_UV"
C_BASELINE_DRIFT_MV = "BASELINE_DRIFT_MV"

C_LEAD_USED = "LEAD_USED"
C_LIMB_SWAP_HYPOTHESIS = "LIMB_SWAP_HYPOTHESIS"


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _mk_code(code: str, text: str, *, system: str = QC_SYSTEM_URI) -> Dict[str, Any]:
    """
    Build a FHIR CodeableConcept with stable coding + human text.
    """
    return {
        "coding": [{"system": system, "code": code}],
        "text": text,
    }


def _mk_quantity(value: float, unit: str, ucum_code: str) -> Dict[str, Any]:
    """
    Build a FHIR Quantity with UCUM coding.
    """
    return {
        "value": float(value),
        "unit": unit,
        "system": UCUM_SYSTEM,
        "code": ucum_code,
    }


def _add_component(
    components: List[Dict[str, Any]],
    *,
    code: str,
    text: str,
    value: Any,
    unit: Optional[str] = None,
    ucum_code: Optional[str] = None,
) -> None:
    """
    Append a FHIR Observation.component entry.

    - Always uses a stable coded CodeableConcept in component.code.
    - If unit+ucum_code are provided, attempts to cast value to float and stores valueQuantity.
    - Otherwise stores valueString.
    - None values are ignored.
    """
    if value is None:
        return

    comp: Dict[str, Any] = {"code": _mk_code(code, text)}

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

    Key mappings:
      - status_strict == FAIL => Observation.status = entered-in-error
      - status_strict == PASS/WARN => Observation.status = final
      - interpretation: PASS->N, WARN->A, FAIL->(omitted)

    Product-safe behavior:
      - If strict=FAIL: keep only QC fields + notes.
        Do NOT export physiologic/derived metrics to avoid unsafe reuse.
    """
    # Normalize inputs
    status_strict = str(qc_out.get("status_strict", qc_out.get("status", "UNKNOWN"))).upper().strip()
    status_ds = str(qc_out.get("status_dataset_aligned", "UNKNOWN"))
    reasons = qc_out.get("reasons", []) or []
    warnings = qc_out.get("warnings", []) or []
    metrics = qc_out.get("metrics", {}) or {}

    eff = effective_datetime or _now_iso_z()

    # FHIR status: FAIL -> entered-in-error, else final
    fhir_status = "entered-in-error" if status_strict == "FAIL" else "final"

    # Minimal FHIR Observation
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "status": fhir_status,
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "procedure",
                        "display": "Procedure",
                    }
                ]
            }
        ],
        "code": {
            "coding": [
                {"system": "http://loinc.org", "code": "11502-2", "display": "EKG study"}
            ],
            "text": "12-lead ECG quality control (QC) + derived metrics",
        },
        "effectiveDateTime": eff,
        "issued": _now_iso_z(),
    }

    if observation_id:
        obs["id"] = observation_id

    if patient_id:
        obs["subject"] = {"reference": f"Patient/{patient_id}"}

    if device_id:
        obs["device"] = {"reference": f"Device/{device_id}"}

    # Interpretation (demo triage): PASS -> N, WARN -> A, FAIL -> omitted
    interp_code = "N" if status_strict == "PASS" else ("A" if status_strict == "WARN" else None)
    if interp_code:
        obs["interpretation"] = [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                        "code": interp_code,
                    }
                ]
            }
        ]

    # Components (always include QC meta; include metrics only if not FAIL)
    comps: List[Dict[str, Any]] = []

    _add_component(
        comps,
        code=C_QC_STATUS_STRICT,
        text="QC status (strict)",
        value=status_strict,
    )
    _add_component(
        comps,
        code=C_QC_STATUS_DATASET,
        text="QC status (dataset aligned)",
        value=status_ds,
    )

    allow_metrics = (status_strict != "FAIL")

    if allow_metrics:
        _add_component(
            comps,
            code=C_FS_HZ,
            text="Sampling frequency",
            value=metrics.get("fs_hz"),
            unit="Hz",
            ucum_code="Hz",
        )
        _add_component(
            comps,
            code=C_HR_BPM,
            text="Heart rate",
            value=metrics.get("hr_bpm"),
            unit="beats/min",
            ucum_code="/min",
        )

        _add_component(comps, code=C_PR_MS, text="PR interval (estimated)", value=metrics.get("pr_ms_est"), unit="ms", ucum_code="ms")
        _add_component(comps, code=C_QRS_MS, text="QRS duration (estimated)", value=metrics.get("qrs_ms_est"), unit="ms", ucum_code="ms")
        _add_component(comps, code=C_QT_MS, text="QT interval (estimated)", value=metrics.get("qt_ms_est"), unit="ms", ucum_code="ms")
        _add_component(comps, code=C_QTC_MS, text="QTc (Bazett, estimated)", value=metrics.get("qtc_ms_est"), unit="ms", ucum_code="ms")

        # Ratios: UCUM supports unitless as code "1"
        _add_component(
            comps,
            code=C_SNR_MED,
            text="SNR (median, unitless ratio)",
            value=metrics.get("snr_med"),
            unit="1",
            ucum_code="1",
        )

        _add_component(comps, code=C_NOISE_RMS_UV, text="Noise RMS", value=metrics.get("noise_rms_uv"), unit="uV", ucum_code="uV")
        _add_component(comps, code=C_BASELINE_DRIFT_MV, text="Baseline drift", value=metrics.get("baseline_drift_mv"), unit="mV", ucum_code="mV")

        _add_component(comps, code=C_LEAD_USED, text="Lead used", value=metrics.get("lead_used"))
        _add_component(comps, code=C_LIMB_SWAP_HYPOTHESIS, text="Limb swap hypothesis", value=metrics.get("limb_swap_hypothesis"))

    obs["component"] = comps

    # Notes: warnings + reasons (human-readable)
    note_lines: List[str] = []
    note_lines.append(f"QC strict={status_strict} | benchmark={status_ds}")

    if reasons:
        note_lines.append("Reasons: " + " | ".join([str(r) for r in reasons]))

    if warnings:
        wtxt: List[str] = []
        for w in warnings:
            if isinstance(w, dict):
                code = str(w.get("code", "") or "").strip()
                reason = str(w.get("reason", "") or "").strip()
                sev = str(w.get("severity", "") or "").strip()
                tag = f"{code}" + (f"[{sev}]" if sev else "")
                line = f"{tag}: {reason}".strip()
                if line:
                    wtxt.append(line)
            else:
                wtxt.append(str(w))
        if wtxt:
            note_lines.append("Warnings: " + " | ".join(wtxt))

    if note_lines:
        obs["note"] = [{"text": "\n".join(note_lines)}]

    return obs


# Terminus

