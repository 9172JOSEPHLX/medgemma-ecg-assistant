from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


QCLevel = Literal["PASS", "WARN", "FAIL"]
ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]


class QCInfo(BaseModel):
    qc_level: QCLevel
    qc_score: int = Field(ge=0, le=100)
    reasons: List[str] = []
    metrics: Dict[str, float] = {}
    retake_prompt: str = ""


class RedFlag(BaseModel):
    code: str
    label: str
    rationale: str
    severity: Literal["LOW", "MEDIUM", "HIGH"] = "MEDIUM"


class Finding(BaseModel):
    name: str
    value: str
    rationale: str = ""
    supporting_leads: List[str] = []


class ReportMeta(BaseModel):
    report_version: str = "1.0"
    model_id: str = "medgemma_unknown"
    model_version: str = "unknown"
    device: str = "unknown"
    mode_avion: bool = True
    timestamp_utc: Optional[str] = None


class ECGReport(BaseModel):
    meta: ReportMeta
    qc: QCInfo

    summary: str
    confidence: ConfidenceLevel = "LOW"

    red_flags: List[RedFlag] = []
    findings: List[Finding] = []

    clinician_checklist: List[str] = []

    disclaimer: str = (
        "Aide à l’interprétation/triage uniquement. "
        "Ne constitue pas un diagnostic médical. Validation par clinicien requise."
    )

    debug: Dict[str, Any] = {}
