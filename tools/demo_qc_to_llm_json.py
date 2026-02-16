### tools\demo_qc_to_llm_json.py  ### Feb 14th, 2026 (updated) 20H50

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)

QC_SAMPLE = {
  "status": "WARN",
  "reasons": ["Baseline wander detected"],
  "warnings": [{"code": "WARN_BW", "reason": "baseline wander high", "severity": "medium"}],
  "metrics": {"snr_db": 12.3, "fs_hz": 500}
}

SYSTEM = (
  "You are a clinical-grade assistant for ECG quality control. "
  "Return STRICT JSON only (no markdown, no prose). "
  "Keys: summary, status, key_findings, actions, fhir_notes."
)

REQUIRED_KEYS = {"summary", "status", "key_findings", "actions", "fhir_notes"}

def iter_json_objects(text: str):
  # Find all {...} blocks; try parsing; yield dicts.
  # Use non-greedy match to reduce over-capture.
  for m in re.finditer(r"\{.*?\}", text, flags=re.DOTALL):
    candidate = m.group(0).strip()
    try:
      obj = json.loads(candidate)
      if isinstance(obj, dict):
        yield obj
    except Exception:
      continue

def pick_summary_json(text: str):
  # Prefer the LAST valid JSON object in text that matches REQUIRED_KEYS.
  # (tiny models tend to echo QC first, and the "answer" later if any)
  objs = list(iter_json_objects(text))
  for obj in reversed(objs):
    if REQUIRED_KEYS.issubset(set(obj.keys())):
      return obj
  return None

def rule_based_fallback(qc: dict) -> dict:
  status = qc.get("status", "WARN")
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

def generate_text(model, tok, prompt: str, max_new_tokens=220):
  inputs = tok(prompt, return_tensors="pt").to("cuda")
  out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
  return tok.decode(out[0], skip_special_tokens=True)

def main():
  assert torch.cuda.is_available(), "CUDA not available"

  model_id = "sshleifer/tiny-gpt2"  # toy model; will not reliably output JSON summary
  tok = AutoTokenizer.from_pretrained(model_id)
  model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

  prompt = SYSTEM + "\nQC=" + json.dumps(QC_SAMPLE) + "\nReturn JSON now:\n"
  text = generate_text(model, tok, prompt)

  obj = pick_summary_json(text)
  if obj is None:
    obj = rule_based_fallback(QC_SAMPLE)

  print(json.dumps(obj, indent=2))

if __name__ == "__main__":
  main()

# TERMINUS