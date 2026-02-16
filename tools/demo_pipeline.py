import json
import os
import re
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Demo input (replace later with your real QC output JSON)
# -----------------------------
QC_SAMPLE = {
  "status": "WARN",
  "reasons": ["Baseline wander detected"],
  "warnings": [{"code": "WARN_BW", "reason": "baseline wander high", "severity": "medium"}],
  "metrics": {"snr_db": 12.3, "fs_hz": 500}
}

REQUIRED_KEYS = {"summary", "status", "key_findings", "actions", "fhir_notes"}


# -----------------------------
# Utilities
# -----------------------------
def iter_json_objects(text: str):
  # Find all {...} blocks; try parsing; yield dicts.
  for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
    candidate = m.group(0).strip()
    try:
      obj = json.loads(candidate)
      if isinstance(obj, dict):
        yield obj
    except Exception:
      continue

def pick_summary_json(text: str):
  objs = list(iter_json_objects(text))
  for obj in reversed(objs):
    if REQUIRED_KEYS.issubset(set(obj.keys())):
      return obj
  return None

def rule_based_fallback(qc: dict) -> dict:
  status = (qc.get("status") or "WARN").upper()
  reasons = qc.get("reasons", []) or []
  warns = qc.get("warnings", []) or []
  metrics = qc.get("metrics", {}) or {}

  key_findings = []
  if reasons:
    key_findings.extend([str(r) for r in reasons[:3]])
  if warns:
    key_findings.extend([f"{w.get('code','WARN')}: {w.get('reason','')}".strip() for w in warns[:3]])
  if not key_findings:
    key_findings = ["No issues detected."]

  if status == "PASS":
    actions = ["Proceed to analysis.", "Export FHIR report."]
    summary = "ECG quality is acceptable for downstream processing."
  elif status == "WARN":
    actions = [
      "Review acquisition quality; consider repeating ECG if clinically appropriate.",
      "Apply IEC bandpass 0.05–150 Hz and re-run QC.",
      "If artefacts persist, capture again (patient still, good electrode contact).",
    ]
    summary = "ECG quality is borderline; artefacts detected. Use caution."
  else:
    actions = [
      "Acquisition not usable for automated interpretation.",
      "Repeat capture with correct calibration and stable baseline.",
      "If repeated FAIL, escalate to manual review.",
    ]
    summary = "ECG quality is insufficient; repeat acquisition required."

  fhir_notes = []
  if "snr_db" in metrics:
    fhir_notes.append(f"SNR(dB)={metrics['snr_db']}")
  if "fs_hz" in metrics:
    fhir_notes.append(f"fs_hz={metrics['fs_hz']}")
  if warns:
    fhir_notes.append("QC warnings present; interpretation should be cautious.")
  if not fhir_notes:
    fhir_notes = ["QC summary generated from QC metrics."]

  return {
    "summary": summary,
    "status": status,
    "key_findings": key_findings,
    "actions": actions,
    "fhir_notes": fhir_notes,
    "source": "fallback"
  }

def ui_messages_from_status(status: str):
  s = (status or "WARN").upper()
  if s == "PASS":
    return {
      "badge": "PASS",
      "primary": "Qualité ECG acceptable.",
      "secondary": "Vous pouvez lancer l’analyse et exporter le rapport."
    }
  if s == "FAIL":
    return {
      "badge": "FAIL",
      "primary": "Qualité ECG insuffisante.",
      "secondary": "Reprenez l’acquisition (calibration, stabilité, contact électrodes)."
    }
  return {
    "badge": "WARN",
    "primary": "Qualité ECG limite : artefacts détectés.",
    "secondary": "Analyse possible avec prudence ; idéalement refaire une capture."
  }

def build_fhir_observation_stub(qc: dict, summary: dict) -> dict:
  now = datetime.now(timezone.utc).isoformat()

  # FHIR-like Observation (simplified; later map to proper codes/LOINC)
  obs = {
    "resourceType": "Observation",
    "status": "final",
    "category": [{"text": "ECG Quality Control"}],
    "code": {"text": "ECG QC Summary"},
    "effectiveDateTime": now,
    "valueString": f"{summary.get('status')} - {summary.get('summary')}",
    "note": [{"text": n} for n in (summary.get("fhir_notes") or [])],
    "component": []
  }

  # Add a few metric components if present
  metrics = qc.get("metrics", {}) or {}
  if "snr_db" in metrics:
    obs["component"].append({
      "code": {"text": "snr_db"},
      "valueQuantity": {"value": float(metrics["snr_db"]), "unit": "dB"}
    })
  if "fs_hz" in metrics:
    obs["component"].append({
      "code": {"text": "fs_hz"},
      "valueQuantity": {"value": float(metrics["fs_hz"]), "unit": "Hz"}
    })

  # Add QC status
  obs["component"].append({
    "code": {"text": "qc_status"},
    "valueString": str(qc.get("status", "WARN")).upper()
  })

  return obs

def try_llm_summary(qc: dict) -> dict | None:
  # Uses a tiny toy model to validate infra; replace with MedGemma later.
  # Returns a dict if it successfully extracts required JSON schema; else None.
  if not torch.cuda.is_available():
    return None

  token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
  # Token is optional for public models, but helps rate limits.
  # We do not hard-fail if missing, to keep demo robust.
  model_id = os.getenv("DEMO_MODEL_ID", "sshleifer/tiny-gpt2")

  system = (
    "You are a clinical-grade assistant for ECG quality control. "
    "Return STRICT JSON only (no markdown, no prose). "
    "Keys: summary, status, key_findings, actions, fhir_notes. "
    "Status must be PASS|WARN|FAIL."
  )

  prompt = system + "\nQC=" + json.dumps(qc) + "\nReturn JSON now:\n"

  tok = AutoTokenizer.from_pretrained(model_id, token=token)
  model = AutoModelForCausalLM.from_pretrained(model_id, token=token).to("cuda")

  inputs = tok(prompt, return_tensors="pt").to("cuda")
  out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
  text = tok.decode(out[0], skip_special_tokens=True)

  obj = pick_summary_json(text)
  return obj


def main():
  qc = QC_SAMPLE

  # 1) LLM attempt
  llm_obj = None
  try:
    llm_obj = try_llm_summary(qc)
  except Exception as e:
    # Keep demo running no matter what
    llm_obj = None

  # 2) Fallback if needed
  if llm_obj is None:
    summary = rule_based_fallback(qc)
  else:
    llm_obj["source"] = "llm"
    summary = llm_obj

  # 3) UI messages
  ui = ui_messages_from_status(summary.get("status", qc.get("status", "WARN")))

  # 4) FHIR stub
  fhir = build_fhir_observation_stub(qc, summary)

  # 5) Print demo outputs (ready for UI screen #2 + export)
  print("=== UI_BADGE ===")
  print(ui["badge"])
  print("\n=== UI_PRIMARY_MESSAGE ===")
  print(ui["primary"])
  print("\n=== UI_SECONDARY_MESSAGE ===")
  print(ui["secondary"])

  print("\n=== SUMMARY_JSON ===")
  print(json.dumps(summary, indent=2, ensure_ascii=False))

  print("\n=== FHIR_OBSERVATION_STUB ===")
  print(json.dumps(fhir, indent=2, ensure_ascii=False))


if __name__ == "__main__":
  main()
