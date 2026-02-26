# qc.py  Version du 06.02.2026 ENRICHI DES BLOCS “QC Normes/Clinique/Mesures”

from __future__ import annotations
from medgem_poc.resample_to_500 import resample_leads_to_500hz, TARGET_FS_HZ
# NOTE: resampling helper lives at:
# from medgem_poc.resample_to_500 import resample_leads_to_500hz, TARGET_FS_HZ


from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple, Literal, Mapping
import json
import math
import csv
from datetime import datetime, timezone

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # image-QC désactivé si OpenCV absent


# =============================================================================
# 0) QC SIGNAL CONSTANTES / "CLINICAL" (durci) — PASS/WARN/FAIL + flags code+reason
# =============================================================================

QC_VERSION = "1.1.1"

Status = Literal["PASS", "WARN", "FAIL"]
_SEVERITY: Dict[Status, int] = {"PASS": 0, "WARN": 1, "FAIL": 2}

# ECG ref
SPEED_MM_S = 25
GAIN_MM_MV = 10
SMALL_SQUARE_MS = 40
LARGE_SQUARE_MS = 200

# QT/QTc thresholds (simplified, pragmatic)
QTc_BORDERLINE = {"M": (430, 450), "F": (450, 460)}
QTc_PROLONGED = {"M": 450, "F": 460}
QTc_HIGH_RISK = 500
QTc_SHORT = 390
QRS_WIDE = 120  # ms


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _escalate(cur: Status, new: Status) -> Status:
    return new if _SEVERITY[new] > _SEVERITY[cur] else cur


def _worst_status(a: Optional[Status], b: Optional[Status]) -> Status:
    a0: Status = a if a in ("PASS", "WARN", "FAIL") else "PASS"
    b0: Status = b if b in ("PASS", "WARN", "FAIL") else "PASS"
    return a0 if _SEVERITY[a0] >= _SEVERITY[b0] else b0


def _flag(*args):
    """
    Backward/forward compatible helper.

    Supported call patterns:
      - _flag(warnings, code, reason)
      - _flag(warnings, code, reason, severity)
      - _flag(warnings, reasons, code, reason)
      - _flag(warnings, reasons, code, reason, severity)

    Where:
      - warnings: list of dicts (at least {code, reason}, optional severity)
      - reasons: list of codes (optional), if provided we ensure code ∈ reasons
      - severity: "PASS"|"WARN"|"FAIL" (optional; if present stored in warning dict)
    """
    if len(args) == 3:
        warnings, code, reason = args
        reasons = None
        severity = None
    elif len(args) == 4:
        # ambiguous: could be (warnings, code, reason, severity) OR (warnings, reasons, code, reason)
        warnings = args[0]
        if isinstance(args[1], list):
            reasons, code, reason = args[1], args[2], args[3]
            severity = None
        else:
            code, reason, severity = args[1], args[2], args[3]
            reasons = None
    elif len(args) == 5:
        warnings, reasons, code, reason, severity = args
    else:
        raise TypeError(f"_flag() expected 3/4/5 args, got {len(args)}")

    w = {"code": str(code), "reason": str(reason)}
    if severity is not None:
        w["severity"] = str(severity)
    warnings.append(w)

    if reasons is not None:
        try:
            if code not in reasons:
                reasons.append(code)
        except Exception:
            pass

# ----------------------------------------------------------------------
# Clinical warnings: visible but do NOT degrade technical QC gate
# ----------------------------------------------------------------------
CLINICAL_WARN_CODES = {
    "WARN_QTC_SHORT",
    "WARN_QTC_PROLONGED",
    "WARN_QTC_HIGH_RISK",
    "WARN_QRS_WIDE",
}


def compute_strict(status: str, warnings: List[Dict[str, Any]]) -> str:
    """
    Strict QC gate = signal quality only.
    Clinical warnings do NOT degrade strict status.
    FAIL always wins if any technical FAIL_* appears in warnings.
    """
    # Hard fail always wins
    if str(status).upper() == "FAIL":
        return "FAIL"

    tech_warn = False

    for w in warnings or []:
        if not isinstance(w, dict):
            continue

        code = str(w.get("code", "")).strip()
        sev = str(w.get("severity", "")).upper().strip()

        if not code:
            continue

        # Ignore clinical warnings in strict gate
        if code in CLINICAL_WARN_CODES:
            continue

        # Any technical FAIL should fail strict gate
        if code.startswith("FAIL_") or sev == "FAIL":
            return "FAIL"

        # Any technical WARN should downgrade to WARN
        if code.startswith("WARN_") or sev == "WARN":
            tech_warn = True

    return "WARN" if tech_warn else "PASS"



def _compute_status_strict_from_warnings(warnings: List[Dict[str, Any]]) -> str:
    """
    Strict QC gate = signal quality only.
    Clinical warnings do NOT degrade strict status.
    FAIL always wins if any technical FAIL_* appears.
    """
    status = "PASS"

    for w in warnings or []:
        if not isinstance(w, dict):
            continue

        code = str(w.get("code", "")).strip()
        severity = str(w.get("severity", "")).upper().strip()

        if not code:
            continue

        if code in CLINICAL_WARN_CODES:
            continue

        if code.startswith("FAIL_") or severity == "FAIL":
            return "FAIL"

        if code.startswith("WARN_") or severity == "WARN":
            status = "WARN"

    return status




def _normalize_sex(patient_sex: Any) -> str:
    s = str(patient_sex).strip().upper() if patient_sex is not None else "M"
    return "F" if s.startswith("F") else "M"


def calc_qtc_bazett(qt_ms: float, rr_ms: float) -> Optional[float]:
    """Bazett QTc (ms) = QT(ms) / sqrt(RR(s))."""
    if not (_is_finite(qt_ms) and _is_finite(rr_ms)):
        return None
    rr_s = float(rr_ms) / 1000.0
    if rr_s <= 0:
        return None
    return float(qt_ms) / math.sqrt(rr_s)

###### ADD FEB 21 TH 2026 10H30 ###### 

import numpy as np

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    n = min(a.size, b.size)
    if n < 50:
        return 0.0
    a = a[:n] - float(np.mean(a[:n]))
    b = b[:n] - float(np.mean(b[:n]))
    da = float(np.std(a)) + 1e-6
    db = float(np.std(b)) + 1e-6
    return float(np.mean((a/da) * (b/db)))

def _apply_limb_swap(leads, kind: str):
    I, II, III = leads["I"], leads["II"], leads["III"]
    aVR, aVL, aVF = leads["aVR"], leads["aVL"], leads["aVF"]

    if kind == "RA_LA":
        return {"I": -I, "II": III, "III": II, "aVR": aVL, "aVL": aVR, "aVF": aVF}
    if kind == "RA_LL":
        return {"I": -III, "II": -II, "III": -I, "aVR": aVF, "aVL": aVL, "aVF": aVR}
    if kind == "LA_LL":
        return {"I": II, "II": I, "III": -III, "aVR": aVR, "aVL": aVF, "aVF": aVL}

    return {"I": I, "II": II, "III": III, "aVR": aVR, "aVL": aVL, "aVF": aVF}

def detect_limb_reversal(leads):
    required = ["I","II","III","aVR","aVL","aVF","V6"]
    if any(k not in leads for k in required):
        return {"kind": "NONE", "score_best": 0.0, "score_none": 0.0, "delta": 0.0}

    V6 = leads["V6"]
    V5 = leads.get("V5", None)

    def score(limb):
        # key heuristic: after correction, I tends to resemble V6 (lateral)
        s = max(0.0, _corr(limb["I"], V6))
        s += 0.5 * max(0.0, _corr(limb["aVL"], V6))
        if V5 is not None:
            s += 0.25 * max(0.0, _corr(limb["I"], V5))
        return float(s)

    none = _apply_limb_swap(leads, "NONE")
    score_none = score(none)

    best_kind, best_score = "NONE", score_none
    for kind in ("RA_LA", "RA_LL", "LA_LL"):
        limb = _apply_limb_swap(leads, kind)
        sc = score(limb)
        if sc > best_score:
            best_kind, best_score = kind, sc

    return {
        "kind": best_kind,
        "score_best": float(best_score),
        "score_none": float(score_none),
        "delta": float(best_score - score_none),
    }

def qc_signal(metrics: Dict[str, Any], patient_sex: Any = "M") -> Dict[str, Any]:
    """
    Signal QC / clinical-ish heuristics.
    Returns:
      {
        'status': 'PASS'|'WARN'|'FAIL',
        'flags': [ {code, reason, severity?}, ... ],
        'derived': {...},
        'metrics': metrics (possibly with rr_ms filled)
      }
    """
    flags: List[Dict[str, str]] = []
    status: Status = "PASS"
    sex = _normalize_sex(patient_sex)

    # RR fallback from HR if missing
    hr = metrics.get("hr_bpm")
    rr = metrics.get("rr_ms")
    if not _is_finite(rr) and _is_finite(hr) and float(hr) > 0:
        rr = 60000.0 / float(hr)
        metrics = dict(metrics)
        metrics["rr_ms"] = rr

    # 1) Signal quality: SNR + baseline drift
    snr = metrics.get("snr") if "snr" in metrics else metrics.get("snr_med")
    if _is_finite(snr):
        snr_f = float(snr)
        if snr_f < 5:
            _flag(
                flags,
                "FAIL_LOW_SNR",
                "SNR too low; extracted signal likely unreliable.",
                "FAIL",
            )
            status = _escalate(status, "FAIL")
        elif snr_f < 10:
            _flag(flags, "WARN_LOW_SNR", "SNR marginal; review recommended.", "WARN")
            status = _escalate(status, "WARN")
    else:
        _flag(
            flags,
            "WARN_SNR_MISSING",
            "SNR not provided; cannot fully assess signal quality.",
            "WARN",
        )
        status = _escalate(status, "WARN")

    baseline = metrics.get("baseline_drift") if "baseline_drift" in metrics else metrics.get("baseline_drift_mv")
    if _is_finite(baseline) and float(baseline) > 0.5:
        _flag(
            flags,
            "WARN_BASELINE_DRIFT",
            "Baseline wander/d drift appears elevated.",
            "WARN",
        )
        status = _escalate(status, "WARN")

    # 2) Filter suspicion
    filt = metrics.get("filter_strength")
    if _is_finite(filt) and float(filt) > 0.8:
        _flag(
            flags,
            "WARN_FILTER_APPLIED",
            "Strong filtering suspected; QRS amplitude/morphology may be altered.",
            "WARN",
        )
        status = _escalate(status, "WARN")

    # 3) QRS width
    qrs = metrics.get("qrs_ms") if "qrs_ms" in metrics else metrics.get("qrs_ms_est")
    if _is_finite(qrs) and float(qrs) >= QRS_WIDE:
        _flag(flags, "WARN_QRS_BROAD", f"QRS is wide ({float(qrs):.0f} ms).", "WARN")
        _flag(
            flags,
            "WARN_QTC_QRS_ADJUST",
            "QTc not adjusted for wide QRS (interpret with caution).",
            "WARN",
        )
        status = _escalate(status, "WARN")

    # 4) QT and QTc (Bazett)
    qt = metrics.get("qt_ms") if "qt_ms" in metrics else metrics.get("qt_ms_est")
    rr_ms = metrics.get("rr_ms")
    derived: Dict[str, Any] = {}

    qtc = (
        calc_qtc_bazett(float(qt), float(rr_ms))
        if (_is_finite(qt) and _is_finite(rr_ms))
        else None
    )
    if qtc is not None:
        derived["qtc_bazett_ms"] = float(qtc)

        if qtc >= QTc_HIGH_RISK:
            _flag(
                flags,
                "WARN_QTC_HIGH_RISK",
                f"QTc ≥ {QTc_HIGH_RISK} ms ({qtc:.0f} ms).",
                "WARN",
            )
            status = _escalate(status, "WARN")
        elif qtc >= QTc_PROLONGED[sex]:
            _flag(
                flags,
                "WARN_QTC_PROLONGED",
                f"QTc prolonged ({qtc:.0f} ms) for sex={sex}.",
                "WARN",
            )
            status = _escalate(status, "WARN")
        elif qtc <= QTc_SHORT:
            _flag(flags, "WARN_QTC_SHORT", f"QTc short ({qtc:.0f} ms).", "WARN")
            status = _escalate(status, "WARN")
    elif _is_finite(qt) and not _is_finite(rr_ms):
        _flag(
            flags,
            "WARN_QTC_UNCOMPUTABLE",
            "QT provided but RR missing; QTc cannot be computed.",
            "WARN",
        )
        status = _escalate(status, "WARN")

    # 5) Electrode inversion heuristics (scalar-feature based)
    leads = metrics.get("lead_signals") or {}
    try:
        i = leads.get("I") or leads.get("DI")
        ii = leads.get("II") or leads.get("DII")
        iii = leads.get("III") or leads.get("DIII")
        avr = leads.get("aVR")

        if _is_finite(iii) and _is_finite(avr):
            if float(iii) < -0.2 and float(avr) > 0:
                _flag(
                    flags,
                    "WARN_BG_JG_INVERSION",
                    "Possible limb electrode inversion (BG↔JG / LA↔LL).",
                    "WARN",
                )
                status = _escalate(status, "WARN")

        if _is_finite(i) and _is_finite(avr):
            if float(i) < -0.2 and float(avr) > 0.1:
                _flag(
                    flags,
                    "WARN_RA_LA_REVERSAL",
                    "Possible RA↔LA limb lead reversal (Lead I inverted, aVR positive).",
                    "WARN",
                )
                status = _escalate(status, "WARN")

        if _is_finite(i) and _is_finite(ii) and _is_finite(iii):
            if (
                abs(float(i)) < 0.05
                and abs(float(ii)) < 0.05
                and abs(float(iii)) < 0.05
            ):
                _flag(
                    flags,
                    "WARN_LIMB_LEADS_LOW",
                    "Limb leads appear very low amplitude; possible poor contact / misplacement.",
                    "WARN",
                )
                status = _escalate(status, "WARN")

    except Exception:
        _flag(
            flags,
            "WARN_ELECTRODE_RULES_ERROR",
            "Electrode inversion heuristics could not be evaluated.",
            "WARN",
        )
        status = _escalate(status, "WARN")

    # Final: any FAIL_* forces FAIL
    if any(f.get("severity") == "FAIL" or str(f.get("code", "")).startswith("FAIL") for f in flags):
        status = "FAIL"

    return {
        "status": status,
        "flags": flags,
        "derived": derived,
        "metrics": metrics,
    }


# =============================================================================
# 1) IMAGE QC (ton code) — on ajoute flags textuels (code+reason+severity)
# =============================================================================


@dataclass
class QCThresholds:
    # Occlusion / stains / folds
    black_blob_fail_ratio: float = 0.020
    black_blob_warn_ratio: float = 0.010
    shadow_fail_ratio: float = 0.35
    shadow_warn_ratio: float = 0.20

    # Severe crease / fold: combined condition (safer than lowering global fail too much)
    crease_fail_shadow: float = 0.23
    crease_fail_black: float = 0.014

    # Valeurs de départ “safe”. On les ajustera après un mini-batch terrain de 50–100 images.
    warp_warn_deg: float = 1.0
    warp_fail_deg: float = 2.5
    warp_evidence_min: int = 8

    # M1 Blur (Var(Laplacian))
    blur_fail: float = 60.0
    blur_warn: float = 120.0

    # M2 Exposure / contrast
    p_white_fail: float = 0.85
    p_black_fail: float = 0.20
    contrast_fail: float = 35.0
    contrast_warn: float = 55.0

    # M3 Glare blobs
    glare_fail_ratio: float = 0.008  # 0.8%
    glare_warn_ratio: float = 0.003  # 0.3%

    # M4 Skew
    skew_warn_deg: float = 7.0
    skew_fail_deg: float = 12.0

    # M5 Framing / document fill
    doc_fill_fail: float = 0.55
    doc_fill_warn: float = 0.70

    # M5 ink density
    ink_ratio_fail: float = 0.010


@dataclass
class QCResult:
    qc_level: str  # "PASS" | "WARN" | "FAIL"
    qc_score: int  # 0..100 (informational)
    metrics: Dict[str, float]
    reasons: List[str]  # codes only (compat)
    retake_prompt: str

    # New: structured flags (code + reason + severity)
    flags: List[Dict[str, str]] = field(default_factory=list)

    # New: allow attaching signal QC report (optional)
    signal_qc: Optional[Dict[str, Any]] = None

    mode_avion_expected: bool = True
    fail_type: str = "NONE"


_REASON_TEXT: Dict[str, str] = {
    "WARP_FAIL": "Strong non-projective warp/curvature detected (grid residual too high).",
    "WARP_WARN": "Possible warp/curvature detected (grid residual elevated).",
    "OCCLUSION_FAIL": "Occlusion/fold/shadow detected (large dark blob and/or shadow area).",
    "OCCLUSION_WARN": "Possible occlusion/fold/shadow detected; review recommended.",
    "BLUR_FAIL": "Image too blurry (low Laplacian variance).",
    "BLUR_WARN": "Image slightly blurry; review recommended.",
    "EXPOSURE_FAIL": "Exposure problem detected (over/under-exposed with low usable contrast).",
    "CONTRAST_FAIL": "Very low contrast; ECG trace may be unreadable.",
    "CONTRAST_WARN": "Low contrast; ECG trace may be degraded.",
    "GLARE_FAIL": "Strong glare/reflection detected (large saturated blob).",
    "GLARE_WARN": "Glare/reflection detected; may hide details.",
    "DOCFILL_FAIL": "Document fills too little of the frame; capture likely incomplete.",
    "DOCFILL_WARN": "Document fill ratio is low; consider recapturing with better framing.",
    "INK_FAIL": "Very low ink density; trace/grid may be too faint.",
    "SKEW_FAIL": "Severe skew detected; capture not aligned (and no confident document quad).",
    "SKEW_WARN": "Skew detected; consider recapturing with better alignment.",
}


def _severity_from_reason_code(code: str) -> Status:
    return "FAIL" if code.endswith("_FAIL") else "WARN"


def _reasons_to_flags(reasons: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in reasons:
        sev = _severity_from_reason_code(r)
        out.append({"code": r, "reason": _REASON_TEXT.get(r, r), "severity": sev})
    return out


# ======================= HELPERS=======================================

def _extract_long_lines(gray: np.ndarray, orientation: str, k: int = 45) -> np.ndarray:
    """Extract long straight-ish lines using morphology. orientation: 'h' or 'v'."""
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    inv = 255 - g

    if orientation == "h":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k))

    lines = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    _, bw = cv2.threshold(lines, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _hough_angles_deg(bw: np.ndarray, expect: str) -> np.ndarray:
    """Hough lines on binary image. expect 'h' (0°) or 'v'(90°). Returns deviations in degrees."""
    if bw is None or bw.size == 0:
        return np.array([], dtype=np.float32)

    h, w = bw.shape[:2]
    edges = cv2.Canny(bw, 50, 150, apertureSize=3)

    min_len = int(0.18 * min(h, w))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=50, minLineLength=min_len, maxLineGap=25
    )
    if lines is None:
        return np.array([], dtype=np.float32)

    devs = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        if dx == 0.0 and dy == 0.0:
            continue
        ang = math.degrees(math.atan2(dy, dx))
        ang = (ang + 90.0) % 180.0 - 90.0

        if expect == "h":
            dev = abs(ang)
        else:
            dev = abs(90.0 - abs(ang))

        if dev < 25.0:
            devs.append(dev)

    if not devs:
        return np.array([], dtype=np.float32)

    return np.array(devs, dtype=np.float32)


def _warp_grid_residual_deg(gray_w: np.ndarray) -> Tuple[float, int]:
    """
    Mesure un résiduel de warp/courbure via la grille (Hough sur lignes longues).
    Retourne (warp_residual_deg, warp_evidence_n).
    """
    if gray_w is None or gray_w.size == 0:
        return 0.0, 0

    g = gray_w
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    if max(g.shape[:2]) > 900:
        g = _resize_long_side(g, long_side=900)

    bw_h = _extract_long_lines(g, "h", k=31)
    bw_v = _extract_long_lines(g, "v", k=31)

    dh = _hough_angles_deg(bw_h, expect="h")
    dv = _hough_angles_deg(bw_v, expect="v")

    evidence_n = int(dh.size + dv.size)
    if evidence_n < 8:
        return 0.0, evidence_n

    ph = float(np.percentile(dh, 95.0)) if dh.size else 0.0
    pv = float(np.percentile(dv, 95.0)) if dv.size else 0.0
    resid = max(ph, pv)

    return float(resid), evidence_n


def _warp_grid_residual_deg_safe(gray_w: np.ndarray) -> Tuple[float, int]:
    """Compat: accepte float ou (float,int)."""
    ret = _warp_grid_residual_deg(gray_w)
    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
        return float(ret[0]), int(ret[1])
    return float(ret), 0


def _resize_long_side(img: np.ndarray, long_side: int = 768) -> np.ndarray:
    h, w = img.shape[:2]
    s = float(long_side) / float(max(h, w))
    if s >= 1.0:
        return img
    nh, nw = int(round(h * s)), int(round(w * s))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _percentiles(gray: np.ndarray, p1: float, p2: float) -> Tuple[float, float]:
    a = np.percentile(gray, p1)
    b = np.percentile(gray, p2)
    return float(a), float(b)


def _var_laplacian(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _doc_quad(gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Returns (quad_pts, fill_ratio). quad_pts shape (4,2) or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0

    h, w = gray.shape[:2]
    img_area = float(h * w)

    best_quad = None
    best_area = 0.0

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        area = float(cv2.contourArea(approx))
        if area < 0.25 * img_area:
            continue
        if area > best_area:
            best_area = area
            best_quad = approx.reshape(4, 2)

    if best_quad is None:
        return None, 0.0

    fill = best_area / img_area
    return best_quad.astype(np.float32), float(fill)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


def _warp_to_rect(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    quad = _order_quad(quad)
    tl, tr, br, bl = quad

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = int(round(max(wA, wB)))
    maxH = int(round(max(hA, hB)))
    maxW = max(maxW, 1)
    maxH = max(maxH, 1)

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped


def _estimate_skew_deg(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 140)
    if lines is None:
        return 0.0

    angles = []
    for i in range(min(len(lines), 30)):
        rho, theta = lines[i][0]
        deg = (theta * 180.0 / np.pi) - 90.0
        while deg < -45:
            deg += 90
        while deg > 45:
            deg -= 90
        angles.append(deg)

    if not angles:
        return 0.0
    return float(np.median(np.abs(angles)))


def _ink_ratio(gray: np.ndarray) -> float:
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7
    )
    ink = (thr == 0).astype(np.uint8)
    return float(ink.mean())


def _glare_blob_ratio(gray: np.ndarray) -> float:
    mask = (gray > 245).astype(np.uint8) * 255
    if mask.mean() < 1e-6:
        return 0.0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    h, w = gray.shape[:2]
    img_area = float(h * w)
    max_area = 0.0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a > max_area:
            max_area = a
    return max_area / img_area


def _black_blob_ratio(gray: np.ndarray) -> float:
    """
    Détecte une grosse tache sombre (pli / occlusion sombre).
    Grid-proof: enlève traits fins (grille/tracé), puis mesure grosse composante sombre.
    """
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    h, w = g.shape[:2]
    img_area = float(h * w)

    dark_thr = 25
    p1 = float(np.percentile(g, 1.0))
    dark_thr = int(min(45, max(dark_thr, p1 + 5)))

    mask = (g < dark_thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=1
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0.0

    min_area = int(0.002 * img_area)
    max_area = 0
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area and a > max_area:
            max_area = a

    return float(max_area) / img_area


def _shadow_area_ratio(gray: np.ndarray) -> float:
    """
    Proxy ombres/illumination non uniforme, robuste aux grilles.
    Low-freq blur + MAD threshold, ignore border.
    """
    g = gray
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)

    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=25, sigmaY=25).astype(np.float32)
    h, w = bg.shape[:2]
    y0, y1 = int(0.05 * h), int(0.95 * h)
    x0, x1 = int(0.05 * w), int(0.95 * w)
    roi = bg[y0:y1, x0:x1]

    med = float(np.median(roi))
    dev = np.abs(roi - med)

    mad = float(np.median(np.abs(dev - np.median(dev)))) + 1e-6
    thr = max(25.0, 6.0 * mad)

    mask = (dev > thr).astype(np.uint8)
    return float(mask.mean())


def _retake_message(reasons: List[str]) -> str:
    tips = []
    if any(r.startswith("WARP_") for r in reasons):
        tips.append(
            "Évite la courbure/perspective : photo bien au-dessus (90°), recule un peu, et assure les 4 coins visibles."
        )
    if any(r.startswith("BLUR_") for r in reasons):
        tips.append(
            "Stabilise le téléphone, tape pour focus, recule légèrement, évite le zoom numérique."
        )
    if any(r.startswith("EXPOSURE_") for r in reasons) or any(
        r.startswith("CONTRAST_") for r in reasons
    ):
        tips.append(
            "Utilise une lumière uniforme (pas de contre-jour), ajuste l’exposition, évite les zones cramées."
        )
    if any(r.startswith("GLARE_") for r in reasons):
        tips.append(
            "Évite les reflets : change l’angle, désactive le flash, éloigne les sources lumineuses."
        )
    if any(r.startswith("DOCFILL_") for r in reasons) or any(
        r.startswith("INK_") for r in reasons
    ):
        tips.append(
            "Recadre toute la feuille/écran (4 coins visibles), rapproche-toi pour remplir le cadre."
        )
    if any(r.startswith("SKEW_") for r in reasons):
        tips.append(
            "Aligne la feuille avec le cadre, photo bien perpendiculaire (90°)."
        )
    if not tips:
        tips = [
            "Reprends la photo avec un cadrage complet, net, sans reflets, et bonne lumière."
        ]
    return " ".join(tips[:2])


def qc_gate(
    image_bgr: np.ndarray, thresholds: QCThresholds = QCThresholds()
) -> QCResult:
    reasons: List[str] = []
    img = _resize_long_side(image_bgr, long_side=768)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()

    quad, doc_fill = _doc_quad(gray)
    quad_conf = 1.0 if quad is not None else 0.0

    if quad is not None:
        img_w = _warp_to_rect(img, quad)
        gray_w = cv2.cvtColor(img_w, cv2.COLOR_BGR2GRAY)
    else:
        gray_w = gray

    blur = _var_laplacian(gray_w)

    p5, p95 = _percentiles(gray_w, 5, 95)
    contrast = float(p95 - p5)
    p_white = float(np.mean(gray_w > 245))
    p_black = float(np.mean(gray_w < 10))

    glare_ratio = _glare_blob_ratio(gray_w)
    skew_deg = _estimate_skew_deg(gray_w)
    ink = _ink_ratio(gray_w)
    black_blob_ratio = _black_blob_ratio(gray_w)
    shadow_area_ratio = _shadow_area_ratio(gray_w)

    # --- WARP/CURL ---
    warp_residual_deg, warp_evidence_n = _warp_grid_residual_deg_safe(gray_w)
    warp_axis_bias_deg = 0.0

    metrics = {
        "blur_lap_var": float(blur),
        "p_white": float(p_white),
        "p_black": float(p_black),
        "contrast_p95_p5": float(contrast),
        "max_glare_blob_ratio": float(glare_ratio),
        "skew_deg": float(skew_deg),
        "doc_fill": float(doc_fill),
        "ink_ratio": float(ink),
        "quad_confidence": float(quad_conf),
        "black_blob_ratio": float(black_blob_ratio),
        "shadow_area_ratio": float(shadow_area_ratio),
        "warp_axis_bias_deg": float(warp_axis_bias_deg),
        "warp_residual_deg": float(warp_residual_deg),
        "warp_evidence_n": float(warp_evidence_n),
    }

    # --- FAIL rules ---
    if (
        warp_evidence_n >= thresholds.warp_evidence_min
        and warp_residual_deg >= thresholds.warp_fail_deg
    ):
        reasons.append("WARP_FAIL")

    if (shadow_area_ratio >= thresholds.crease_fail_shadow) and (
        black_blob_ratio >= thresholds.crease_fail_black
    ):
        reasons.append("OCCLUSION_FAIL")

    if (black_blob_ratio > thresholds.black_blob_fail_ratio) or (
        shadow_area_ratio > thresholds.shadow_fail_ratio
    ):
        reasons.append("OCCLUSION_FAIL")
    if blur < thresholds.blur_fail:
        reasons.append("BLUR_FAIL")

    exposure_fail = (
        (p_white > thresholds.p_white_fail) and (contrast < thresholds.contrast_warn)
    ) or ((p_black > thresholds.p_black_fail) and (contrast < thresholds.contrast_warn))
    if exposure_fail:
        reasons.append("EXPOSURE_FAIL")
    if contrast < thresholds.contrast_fail:
        reasons.append("CONTRAST_FAIL")
    if glare_ratio > thresholds.glare_fail_ratio:
        reasons.append("GLARE_FAIL")
    if doc_fill > 0.0 and doc_fill < thresholds.doc_fill_fail:
        reasons.append("DOCFILL_FAIL")
    if ink < thresholds.ink_ratio_fail:
        reasons.append("INK_FAIL")
    if (skew_deg > thresholds.skew_fail_deg) and (quad_conf < 0.5):
        reasons.append("SKEW_FAIL")

    # --- WARN rules (only if no FAIL) ---
    has_fail = any(r.endswith("_FAIL") for r in reasons)
    if not has_fail:
        if warp_residual_deg >= thresholds.warp_warn_deg:
            reasons.append("WARP_WARN")

        if (black_blob_ratio > thresholds.black_blob_warn_ratio) or (
            shadow_area_ratio > thresholds.shadow_warn_ratio
        ):
            reasons.append("OCCLUSION_WARN")
        if blur < thresholds.blur_warn:
            reasons.append("BLUR_WARN")
        if contrast < thresholds.contrast_warn:
            reasons.append("CONTRAST_WARN")
        if glare_ratio > thresholds.glare_warn_ratio:
            reasons.append("GLARE_WARN")
        if doc_fill > 0.0 and (
            thresholds.doc_fill_fail <= doc_fill < thresholds.doc_fill_warn
        ):
            reasons.append("DOCFILL_WARN")
        if thresholds.skew_warn_deg < skew_deg <= thresholds.skew_fail_deg:
            reasons.append("SKEW_WARN")

    # --- final level ---
    has_fail = any(r.endswith("_FAIL") for r in reasons)
    if has_fail:
        level = "FAIL"
    elif reasons:
        level = "WARN"
    else:
        level = "PASS"

    # --- QC score (informational) ---
    score = 100
    if blur < thresholds.blur_warn:
        score -= 25 if blur >= thresholds.blur_fail else 45

    exposure_suspect = (
        (p_white > thresholds.p_white_fail) and (contrast < thresholds.contrast_warn)
    ) or ((p_black > thresholds.p_black_fail) and (contrast < thresholds.contrast_warn))
    if exposure_suspect:
        score -= 15

    if contrast < thresholds.contrast_warn:
        score -= 10 if contrast >= thresholds.contrast_fail else 25
    if glare_ratio > thresholds.glare_warn_ratio:
        score -= 10 if glare_ratio <= thresholds.glare_fail_ratio else 25
    if doc_fill > 0.0 and doc_fill < thresholds.doc_fill_warn:
        score -= 10 if doc_fill >= thresholds.doc_fill_fail else 25
    if skew_deg > thresholds.skew_warn_deg:
        score -= 10 if skew_deg <= thresholds.skew_fail_deg else 20

    score = int(max(0, min(100, score)))

    if level == "FAIL":
        score = min(score, 60)
    elif level == "WARN":
        score = min(score, 85)

    retake_prompt = ""
    if level in ("FAIL", "WARN"):
        retake_prompt = _retake_message(reasons)

    if level == "FAIL":
        fail_type = (
            "OCCLUSION"
            if any(r.startswith("OCCLUSION_") for r in reasons)
            else "GEOMETRY"
        )
    else:
        fail_type = "NONE"

    flags = _reasons_to_flags(reasons)

    return QCResult(
        qc_level=level,
        qc_score=score,
        metrics=metrics,
        reasons=reasons,
        retake_prompt=retake_prompt,
        flags=flags,
        signal_qc=None,
        mode_avion_expected=True,
        fail_type=fail_type,
    )


# =============================================================================
# 2) Bundle utilitaire — image QC + signal QC + status global
# =============================================================================


def qc_bundle(
    image_bgr: Optional[np.ndarray] = None,
    signal_metrics: Optional[Dict[str, Any]] = None,
    patient_sex: Any = "M",
    thresholds: QCThresholds = QCThresholds(),
) -> Dict[str, Any]:
    """
    Returns a unified report:
      {
        'status': PASS|WARN|FAIL,
        'image': <QCResult as dict> or None,
        'signal': <qc_signal report> or None,
        'flags': combined flags (image+signal)
      }
    """
    image_qc: Optional[QCResult] = None
    signal_qc: Optional[Dict[str, Any]] = None

    if image_bgr is not None:
        image_qc = qc_gate(image_bgr=image_bgr, thresholds=thresholds)

    if signal_metrics is not None:
        signal_qc = qc_signal(signal_metrics, patient_sex=patient_sex)

    status: Status = "PASS"
    if image_qc is not None:
        status = _worst_status(status, image_qc.qc_level)  # type: ignore[arg-type]
    if signal_qc is not None:
        status = _worst_status(status, signal_qc.get("status"))  # type: ignore[arg-type]

    combined_flags: List[Dict[str, str]] = []
    if image_qc is not None:
        combined_flags.extend(image_qc.flags)
    if signal_qc is not None:
        combined_flags.extend(signal_qc.get("flags", []))

    return {
        "status": status,
        "image": asdict(image_qc) if image_qc is not None else None,
        "signal": signal_qc,
        "flags": combined_flags,
    }


# =============================================================================
# 3) JSON helpers (inchangés, + export bundle)
# =============================================================================


def qc_to_json(qc: QCResult) -> str:
    return json.dumps(asdict(qc), ensure_ascii=False, indent=2)


def load_image_bgr(path: str) -> np.ndarray:
    if cv2 is None:
        raise ImportError(
            "cv2 not installed. Install opencv-python-headless to use image QC."
        )
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =============================================================================
# 4) Example CLI (optional)
# =============================================================================
if __name__ == "__main__":
    sample_signal = {
        "hr_bpm": 72,
        "rr_ms": 833,
        "pr_ms": 160,
        "qrs_ms": 92,
        "qt_ms": 400,
        "snr": 12.5,
        "baseline_drift": 0.2,
        "filter_strength": 0.5,
        "lead_signals": {"DIII": -0.3, "aVR": 0.2, "I": 0.1, "II": 0.2},
    }
    print(
        json.dumps(
            qc_signal(sample_signal, patient_sex="M"), indent=2, ensure_ascii=False
        )
    )

# ======================= SIGNAL QC (v1) =======================

_STATUS_RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worse_status(a: str, b: str) -> str:
    return a if _STATUS_RANK.get(a, 0) >= _STATUS_RANK.get(b, 0) else b


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return mad


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n == 0:
        return x.copy()
    win = int(max(1, win))
    if win <= 1:
        return x.copy()

    # force odd window so that SAME-length result is guaranteed
    if win % 2 == 0:
        win += 1

    k = np.ones((win,), dtype=np.float32) / float(win)

    # use "same" to keep length; then edge-correct with reflect padding
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    y = np.convolve(xp, k, mode="valid")  # len == n (because win is odd)
    if y.size != n:
        # absolute safety: crop or pad to exact length
        if y.size > n:
            y = y[:n]
        else:
            y = np.pad(y, (0, n - y.size), mode="edge")
    return y.astype(np.float32)


# =============================================================================
# IEC NORMALISATION (CSV pipeline) — 0.05–150 Hz + notch optionnel (50/60)
# =============================================================================

def iec_bandpass_filter(
    x: np.ndarray,
    fs: float,
    *,
    notch_hz: Optional[float] = None,
    order: int = 4,
    mode: str = "diagnostic",
    return_metrics: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """
    IEC-ish normalization wrapper (no SciPy).
    Delegates actual DSP to `iec_bandpass_filter_1d`.

    Returns:
      y: filtered signal (float32)
      warnings: list of {code, reason, severity?}
      metrics: dict (always returned; can be empty)

    Notes / future:
      - mode="diagnostic": target 0.05–150 Hz (+ optional notch)
      - mode="monitoring": target 0.67–40 Hz (+ optional notch)
      - future per-lead filtering: accept Mapping[str, np.ndarray] + apply per lead
      - future auto-notch: notch_hz="auto" -> infer 50/60 by region/config
    """
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "iec_filter": True,
        "method": "iec_bandpass_filter_1d",
        "mode": str(mode),
        "fs_hz": float(fs) if _is_finite(fs) else None,
        "order": int(order),
        "notch_hz": float(notch_hz) if _is_finite(notch_hz) else None,
    }

    # Basic input validation (QC-grade, not clinical)
    try:
        x_in = np.asarray(x, dtype=np.float32)
    except Exception:
        _flag(warnings, "FAIL_IEC_FILTER_BAD_INPUT", "Input cannot be converted to float32.", "FAIL")
        metrics["ok"] = False
        metrics["reason"] = "bad_input"
        # return passthrough (safe)
        y = np.asarray(x, dtype=np.float32) if x is not None else np.zeros((0,), dtype=np.float32)
        return y, warnings, metrics

    if x_in.ndim != 1:
        _flag(warnings, "WARN_IEC_FILTER_NDIM", f"Expected 1D signal, got shape={tuple(x_in.shape)}.", "WARN")
        x_in = x_in.reshape(-1).astype(np.float32)

    if not _is_finite(fs) or float(fs) <= 0:
        _flag(warnings, "WARN_IEC_FILTER_FS_INVALID", f"Invalid fs={fs}; using fallback fs=500.", "WARN")
        fs = 500.0
        metrics["fs_hz"] = float(fs)

    if x_in.size < int(max(1, round(float(fs) * 0.5))):
        _flag(warnings, "WARN_IEC_FILTER_TOO_SHORT", "Signal too short for stable filtering; passthrough.", "WARN")
        metrics["ok"] = False
        metrics["reason"] = "too_short"
        return x_in.astype(np.float32), warnings, metrics

    # Call the actual filter (your existing implementation)
    try:
        y, w = iec_bandpass_filter_1d(x_in, float(fs), notch_hz=notch_hz, order=order)
    except TypeError:
        # backward compat if iec_bandpass_filter_1d signature lacks notch/order
        y, w = iec_bandpass_filter_1d(x_in, float(fs))
    except Exception as e:
        _flag(warnings, "FAIL_IEC_FILTER_EXCEPTION", f"IEC filter failed: {type(e).__name__}", "FAIL")
        metrics["ok"] = False
        metrics["reason"] = "exception"
        return x_in.astype(np.float32), warnings, metrics

    y = np.asarray(y, dtype=np.float32)

    # Merge / normalize warnings
    if isinstance(w, list):
        for ww in w:
            if isinstance(ww, dict) and "code" in ww and "reason" in ww:
                warnings.append(ww)
            else:
                # tolerate old warning shapes
                _flag(warnings, "WARN_IEC_FILTER_NOTE", str(ww), "WARN")
    elif w is not None:
        _flag(warnings, "WARN_IEC_FILTER_NOTE", str(w), "WARN")

    # Safety checks on output
    if y.shape != x_in.shape:
        _flag(
            warnings,
            "WARN_IEC_FILTER_LEN_CHANGED",
            f"Filter changed length {y.shape} vs {x_in.shape} (will align).",
            "WARN",
        )
        # align length deterministically
        n = int(x_in.size)
        if y.size > n:
            y = y[:n]
        else:
            y = np.pad(y, (0, n - y.size), mode="edge")

    if not np.isfinite(y).all():
        _flag(warnings, "FAIL_IEC_FILTER_NONFINITE", "Filter output contains NaN/Inf; passthrough.", "FAIL")
        metrics["ok"] = False
        metrics["reason"] = "nonfinite"
        return x_in.astype(np.float32), warnings, metrics

    # “Explosion” detection (QC-grade heuristic)
    in_std = float(np.std(x_in)) + 1e-9
    out_std = float(np.std(y)) + 1e-9
    gain = out_std / in_std
    metrics["std_in"] = float(in_std)
    metrics["std_out"] = float(out_std)
    metrics["std_gain"] = float(gain)

    if gain > 8.0:
        _flag(
            warnings,
            "FAIL_IEC_FILTER_UNSTABLE",
            f"Filter seems unstable (std_gain={gain:.2f}); passthrough.",
            "FAIL",
        )
        metrics["ok"] = False
        metrics["reason"] = "unstable"
        return x_in.astype(np.float32), warnings, metrics

    metrics["ok"] = True

    # Always return 3-tuple (so tools/cts_test.py can consume it cleanly)
    return y, warnings, metrics

# Inserption here Feb 6th, 2026. 3H45PM
# =============================================================================
# IEC-NORMALISATION (CSV pipeline) — 0.05–150 Hz + notch optionnel (50/60)
# SciPy optionnel: fallback 100% numpy (FFT + cosine taper).
# =============================================================================

from typing import Iterable  # si pas déjà importé


def _fft_bandpass_notch(
    x: np.ndarray,
    fs: float,
    *,
    low_hz: float = 0.05,
    high_hz: float = 150.0,
    notch_hz: Optional[float] = None,
    notch_bw_hz: float = 1.0,
    taper_hz: float = 1.0,
) -> np.ndarray:
    """
    NumPy-only frequency-domain bandpass + optional notch.
    - bandpass: keep [low_hz, high_hz] with cosine tapers (reduces ringing).
    - notch: attenuate around notch_hz ± notch_bw_hz/2 (also with taper).
    """
    x = np.asarray(x, dtype=np.float32)
    n = int(x.size)
    if n == 0:
        return x.copy()

    fs = float(fs)
    nyq = 0.5 * fs

    # rfft
    X = np.fft.rfft(x.astype(np.float64), n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    def _cos_taper_mask(freqs: np.ndarray, f0: float, f1: float, taper: float) -> np.ndarray:
        """
        Passband [f0,f1] with cosine ramps of width 'taper' on each side.
        """
        m = np.zeros_like(freqs, dtype=np.float64)

        f0 = max(0.0, float(f0))
        f1 = max(f0, float(f1))
        taper = max(0.0, float(taper))

        if f1 <= 0.0:
            return m

        # flat region
        lo_flat0 = f0 + taper
        hi_flat1 = f1 - taper

        # rising ramp: [f0, f0+taper]
        if taper > 0:
            r0 = f0
            r1 = f0 + taper
            sel = (freqs >= r0) & (freqs < r1)
            # 0..1 raised cosine
            z = (freqs[sel] - r0) / (r1 - r0 + 1e-12)
            m[sel] = 0.5 - 0.5 * np.cos(np.pi * z)
        else:
            lo_flat0 = f0

        # flat pass
        sel = (freqs >= lo_flat0) & (freqs <= hi_flat1) if hi_flat1 >= lo_flat0 else np.zeros_like(freqs, dtype=bool)
        m[sel] = 1.0

        # falling ramp: [f1-taper, f1]
        if taper > 0:
            f0b = max(f0, f1 - taper)
            f1b = f1
            sel = (freqs > f0b) & (freqs <= f1b)
            z = (freqs[sel] - f0b) / (f1b - f0b + 1e-12)
            m[sel] = 0.5 + 0.5 * np.cos(np.pi * z)  # goes 1..0
        else:
            # if no taper, hard cut already handled by flat pass
            pass

        return m

    # bandpass mask
    bp = _cos_taper_mask(freqs, low_hz, min(high_hz, nyq), taper_hz)

    # notch mask (multiplicative attenuator)
    notch = np.ones_like(freqs, dtype=np.float64)
    if notch_hz is not None and notch_hz > 0:
        f0 = float(notch_hz) - float(notch_bw_hz) / 2.0
        f1 = float(notch_hz) + float(notch_bw_hz) / 2.0
        # notch "stop" mask with cosine ramps: 1 outside, 0 inside
        stop = _cos_taper_mask(freqs, max(0.0, f0), min(nyq, f1), taper=max(0.2, 0.25 * notch_bw_hz))
        notch = 1.0 - stop

    Y = X * bp * notch
    y = np.fft.irfft(Y, n=n).astype(np.float32)
    return y


def iec_bandpass_filter_1d(
    x: np.ndarray,
    fs: float,
    *,
    notch_hz: Optional[float] = None,
    order: int = 4,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    IEC-ish preprocessing for digital ECG (QC-only):
      - bandpass: 0.05–150 Hz
      - optional notch: 50/60 Hz (narrow)
    Returns:
      (x_filt, warnings)
    Notes:
      - If SciPy is available, you MAY later swap to true Butterworth (filtfilt).
      - This implementation works without SciPy (FFT+taper fallback).
    """
    warnings: List[Dict[str, str]] = []
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x.copy(), warnings

    if (not _is_finite(fs)) or float(fs) <= 0:
        warnings.append({"code": "WARN_FILTER_BAD_FS", "reason": f"Invalid fs={fs}; IEC filter skipped."})
        return x.copy(), warnings

    fsf = float(fs)
    nyq = 0.5 * fsf

    # validate band
    low_hz = 0.05
    high_hz = 150.0

    if high_hz >= nyq:
        warnings.append(
            {"code": "WARN_FILTER_BAND_CLIPPED", "reason": f"high_hz={high_hz} >= Nyquist={nyq:.1f}; clipped."}
        )
        high_hz = max(0.1, nyq - 0.5)

    if low_hz <= 0 or low_hz >= high_hz:
        warnings.append({"code": "WARN_FILTER_BAD_BAND", "reason": f"Bad band {low_hz}-{high_hz} Hz; skipped."})
        return x.copy(), warnings

    # validate notch
    notch_ok = False
    notch_use = None
    if notch_hz is not None:
        try:
            nhz = float(notch_hz)
            if nhz <= 0 or nhz >= nyq:
                warnings.append(
                    {"code": "WARN_FILTER_NOTCH_INVALID", "reason": f"notch_hz={notch_hz} out of range; ignored."}
                )
            else:
                notch_ok = True
                notch_use = nhz
        except Exception:
            warnings.append({"code": "WARN_FILTER_NOTCH_INVALID", "reason": f"notch_hz={notch_hz} invalid; ignored."})

    # Optional SciPy path (future): keep, but do not require SciPy.
    # If absent, we use NumPy-only fallback.
    use_scipy = False
    try:
        import scipy.signal as sp_signal  # type: ignore
        use_scipy = True
    except Exception:
        use_scipy = False

    if use_scipy:
        # True IIR (Butterworth) + filtfilt; notch via iirnotch (Q~30)
        # Still QC-only.
        try:
            wp = [low_hz / nyq, high_hz / nyq]
            b, a = sp_signal.butter(int(max(2, order)), wp, btype="bandpass")
            y = sp_signal.filtfilt(b, a, x.astype(np.float64)).astype(np.float32)

            if notch_ok and notch_use is not None:
                # narrow notch (Q=30)
                b2, a2 = sp_signal.iirnotch(w0=notch_use / nyq, Q=30.0)
                y = sp_signal.filtfilt(b2, a2, y.astype(np.float64)).astype(np.float32)

            return y, warnings
        except Exception as e:
            warnings.append({"code": "WARN_FILTER_SCIPY_FAILED", "reason": f"SciPy filter failed; fallback FFT. ({e})"})

    # NumPy-only fallback (FFT+taper)
    # notch bandwidth: keep narrow so it doesn't kill QRS
    notch_bw = 1.0  # Hz
    taper = 1.0     # Hz taper at band edges
    y = _fft_bandpass_notch(
        x,
        fsf,
        low_hz=low_hz,
        high_hz=high_hz,
        notch_hz=notch_use if notch_ok else None,
        notch_bw_hz=notch_bw,
        taper_hz=taper,
    )

    if x.size < int(1.0 * fsf):
        warnings.append({"code": "WARN_FILTER_SHORT_SIGNAL", "reason": "Signal < 1s; FFT filtering may be unstable at edges."})

    return y, warnings

# Inserption here Feb 6th, 2026. 3H45PM modified 09H08 PM

from typing import Iterable

def _iec_normalize_leads(
    leads: Mapping[str, np.ndarray],
    fs: float,
    *,
    notch_hz: Optional[float] = 50.0,
    order: int = 4,
    mode: str = "diagnostic",
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """
    Apply IEC-ish bandpass (+ optional notch) to each lead using iec_bandpass_filter().
    Returns:
      filtered_leads: dict lead->filtered 1D float32
      warnings: aggregated warnings (list of dict)
      metrics: aggregated metrics (per-lead + summary)
    """
    # --- notch_hz validation (dataset/demo expectation) ---
    # Accept only 50 or 60 Hz. If invalid, do NOT crash: emit a WARN and skip notch.
    notch_invalid = False
    try:
        if notch_hz is None:
            notch_invalid = False  # explicit disable => OK
        else:
            nhz = float(notch_hz)
            if (not np.isfinite(nhz)) or (nhz not in (50.0, 60.0)):
                notch_invalid = True
    except Exception:
        notch_invalid = True

    filtered: dict[str, np.ndarray] = {}
    warnings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "iec_normalize": True,
        "mode": str(mode),
        "fs_hz": float(fs) if _is_finite(fs) else None,
        "notch_hz": float(notch_hz) if (notch_hz is not None and _is_finite(notch_hz)) else None,
        "order": int(order),
        "per_lead": {},
    }

    if not leads:
        _flag(warnings, "FAIL_IEC_NO_LEADS", "No leads provided to IEC normalize.", "FAIL")
        metrics["ok"] = False
        metrics["reason"] = "no_leads"
        return {}, warnings, metrics

    if notch_invalid:
        warnings.append(
            {
                "code": "WARN_IEC_NOTCH_INVALID",
                "reason": f"Invalid notch_hz={notch_hz}. Expected 50 or 60 Hz. Notch disabled.",
                "severity": "WARN",
            }
        )
        notch_hz = None

    for name, arr in leads.items():
        if arr is None:
            continue
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size < 2:
            continue

        y, w, m = iec_bandpass_filter(
            x, float(fs), notch_hz=notch_hz, order=order, mode=mode, return_metrics=True
        )
        filtered[str(name)] = np.asarray(y, dtype=np.float32)

        # aggregate warnings
        if isinstance(w, list):
            for ww in w:
                if isinstance(ww, dict):
                    warnings.append(ww)
                else:
                    _flag(warnings, "WARN_IEC_FILTER_NOTE", f"{name}: {ww}", "WARN")
        elif w is not None:
            _flag(warnings, "WARN_IEC_FILTER_NOTE", f"{name}: {w}", "WARN")

        # store per-lead metrics (optional)
        if isinstance(m, dict):
            metrics["per_lead"][str(name)] = m

    metrics["n_leads_in"] = int(len(leads))
    metrics["n_leads_out"] = int(len(filtered))
    metrics["ok"] = True if filtered else False
    if not filtered:
        metrics["reason"] = "no_filtered_output"
        _flag(warnings, "FAIL_IEC_NO_OUTPUT", "IEC normalize produced no output leads.", "FAIL")

    return filtered, warnings, metrics


def _robust_snr_and_baseline(x_mv: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    """
    Returns (snr_med, baseline_drift_mv, filter_strength[0..1-ish], noise_rms_uv).
    Heuristics: no clinical guarantee; QC only.
    """
    x = np.asarray(x_mv, dtype=np.float32)
    if x.size < int(fs * 1.0):
        return 0.0, 0.0, 0.0, 0.0

    win_base = int(max(3, round(fs * 0.8)))
    base = _moving_average(x, win_base)
    detr = x - base

    b5 = float(np.percentile(base, 5))
    b95 = float(np.percentile(base, 95))
    baseline_drift = float(b95 - b5)  # mV (robust ~p2p)

    win_smooth = int(max(3, round(fs * 0.04)))
    smooth = _moving_average(detr, win_smooth)
    hf = detr - smooth

    sig_mad = _mad(smooth)
    noise_mad = _mad(hf)
    sig_sigma = float(sig_mad / 0.6745) if sig_mad > 0 else 0.0
    noise_sigma = float(noise_mad / 0.6745) if noise_mad > 0 else 0.0

    snr = float(sig_sigma / (noise_sigma + 1e-6))

    hf_std = float(np.std(hf))
    sm_std = float(np.std(smooth))
    ratio = hf_std / (sm_std + 1e-6)
    filter_strength = float(np.clip((0.08 - ratio) / 0.08, 0.0, 1.0))

    noise_rms_uv = float(noise_sigma * 1000.0)  # mV -> µV

    return snr, baseline_drift, filter_strength, noise_rms_uv


def _detect_r_peaks(x_mv: np.ndarray, fs: float) -> np.ndarray:
    """
    Very simple R-peak detector for QC (not diagnostic).
    Returns indices of peaks.
    """
    x = np.asarray(x_mv, dtype=np.float32)
    if x.size < int(fs * 1.5):
        return np.array([], dtype=np.int32)

    base = _moving_average(x, int(max(3, round(fs * 0.2))))
    y = x - base

    m = float(np.median(y))
    madv = _mad(y)
    s = float(madv / 0.6745) if madv > 0 else float(np.std(y) + 1e-6)
    z = (y - m) / (s + 1e-6)

    thr = 3.0
    refractory = int(round(fs * 0.20))
    cand = np.where(z > thr)[0]
    if cand.size == 0:
        return np.array([], dtype=np.int32)

    peaks = []
    i = 0
    while i < cand.size:
        start = cand[i]
        end = start
        while i < cand.size and cand[i] <= end + refractory:
            end = cand[i]
            i += 1
        seg = y[start : end + 1]
        if seg.size:
            p = int(start + int(np.argmax(seg)))
            peaks.append(p)

    return np.array(peaks, dtype=np.int32)


def _estimate_intervals_ms(
    x_mv: np.ndarray, fs: float
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (qrs_ms_est, qt_ms_est, pr_ms_est) (PR often None).
    Very rough; QC only.
    """
    x = np.asarray(x_mv, dtype=np.float32)
    peaks = _detect_r_peaks(x, fs)
    if peaks.size < 2:
        return None, None, None

    base = _moving_average(x, int(max(3, round(fs * 0.2))))
    y = x - base

    qrs_list = []
    qt_list = []

    for p in peaks[1:-1]:
        w_pre = int(round(fs * 0.12))
        w_post = int(round(fs * 0.45))
        a = max(0, p - w_pre)
        b = min(y.size, p + w_post)
        seg = y[a:b]
        if seg.size < int(round(fs * 0.2)):
            continue

        r_amp = float(np.max(np.abs(seg)))
        if r_amp < 0.05:
            continue
        thr = 0.15 * r_amp

        rp = p - a
        left = seg[:rp] if rp > 3 else seg[:]
        idx_on = None
        for k in range(left.size - 1, -1, -1):
            if abs(float(left[k])) < thr:
                idx_on = k
                break

        right = seg[rp:] if rp < seg.size else np.array([], dtype=np.float32)
        idx_off = None
        for k in range(min(right.size, int(round(fs * 0.2)))):
            if abs(float(right[k])) < thr:
                idx_off = rp + k
                break

        if idx_on is not None and idx_off is not None and idx_off > idx_on:
            qrs_ms = (idx_off - idx_on) * 1000.0 / fs
            qrs_list.append(qrs_ms)

            start_t = idx_off + int(round(fs * 0.06))
            if start_t < seg.size:
                thr_t = 0.08 * r_amp
                stable = int(round(fs * 0.05))
                found = None
                for j in range(
                    start_t, min(seg.size - stable, start_t + int(round(fs * 0.55)))
                ):
                    win = seg[j : j + stable]
                    if np.max(np.abs(win)) < thr_t:
                        if float(np.std(np.diff(win))) < (thr_t * 0.15 + 1e-6):
                            found = j
                            break
                if found is not None:
                    qt_ms = (found - idx_on) * 1000.0 / fs
                    qt_list.append(qt_ms)

    qrs_est = float(np.median(qrs_list)) if qrs_list else None
    qt_est = float(np.median(qt_list)) if qt_list else None
    pr_est = None
    return qrs_est, qt_est, pr_est


def _calc_qtc_bazett(qt_ms: float, rr_ms: float) -> Optional[float]:
    rr_s = rr_ms / 1000.0
    if rr_s <= 0:
        return None
    return float(qt_ms / math.sqrt(rr_s))


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _limb_consistency_error(
    leads: Mapping[str, np.ndarray], fs: float
) -> Optional[float]:
    need = ["I", "II", "III", "aVR", "aVL", "aVF"]
    if not all(k in leads for k in need):
        return None

    I = np.asarray(leads["I"], dtype=np.float32)
    II = np.asarray(leads["II"], dtype=np.float32)
    III = np.asarray(leads["III"], dtype=np.float32)
    aVR = np.asarray(leads["aVR"], dtype=np.float32)
    aVL = np.asarray(leads["aVL"], dtype=np.float32)
    aVF = np.asarray(leads["aVF"], dtype=np.float32)

    n = min(I.size, II.size, III.size, aVR.size, aVL.size, aVF.size)
    if n < int(fs * 1.0):
        return None

    I, II, III, aVR, aVL, aVF = I[:n], II[:n], III[:n], aVR[:n], aVL[:n], aVF[:n]

    e1 = _rms(III - (II - I))
    e2 = _rms(aVR - (-(I + II) / 2.0))
    e3 = _rms(aVL - (I - II / 2.0))
    e4 = _rms(aVF - (II - I / 2.0))

    ref = _rms(II) + 1e-6
    return float((e1 + e2 + e3 + e4) / (4.0 * ref))


def _apply_limb_hypothesis(
    leads: Mapping[str, np.ndarray], hyp: str
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def get(k: str) -> np.ndarray:
        return np.asarray(leads[k], dtype=np.float32)

    if hyp == "NORMAL":
        for k in ["I", "II", "III", "aVR", "aVL", "aVF"]:
            if k in leads:
                out[k] = get(k)
        return out

    if hyp == "LA_LL":
        out["I"] = get("II")
        out["II"] = get("I")
        out["III"] = -get("III")
        out["aVR"] = get("aVR")
        out["aVL"] = get("aVF")
        out["aVF"] = get("aVL")
        return out

    if hyp == "RA_LA":
        out["I"] = -get("I")
        out["II"] = get("III")
        out["III"] = get("II")
        out["aVR"] = get("aVL")
        out["aVL"] = get("aVR")
        out["aVF"] = get("aVF")
        return out

    if hyp == "RA_LL":
        out["I"] = -get("III")
        out["II"] = -get("II")
        out["III"] = -get("I")
        out["aVR"] = get("aVF")
        out["aVL"] = get("aVL")
        out["aVF"] = get("aVR")
        return out

    return _apply_limb_hypothesis(leads, "NORMAL")


def _infer_limb_swap(
    leads: Mapping[str, np.ndarray], fs: float
) -> tuple[Optional[str], Optional[float], Optional[float]]:
    need = ["I", "II", "III", "aVR", "aVL", "aVF"]
    if not all(k in leads for k in need):
        return None, None, None

    normal_err = _limb_consistency_error(leads, fs)
    if normal_err is None:
        return None, None, None

    hyps = ["NORMAL", "LA_LL", "RA_LA", "RA_LL"]
    best_h, best_e = "NORMAL", normal_err
    for h in hyps[1:]:
        mapped = _apply_limb_hypothesis(leads, h)
        e = _limb_consistency_error(mapped, fs)
        if e is not None and e < best_e:
            best_e = e
            best_h = h

    return best_h, best_e, normal_err


def load_leads_from_csv(path: str) -> tuple[dict[str, np.ndarray], float]:
    """
    Robust loader:
    - first row header expected
    - optional 'time' column ignored
    - returns (leads_dict, fs_hz_guess)
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]

        rows = []
        for r in reader:
            if not r:
                continue
            rows.append(r)

    if not rows:
        return {}, 500.0

    arr = np.array(rows, dtype=np.float32)
    cols = {name: i for i, name in enumerate(header)}
    cols_l = {str(name).strip().lower(): i for name, i in cols.items()}

    time_idx = None
    # Many demo CSVs use "time_s" (seconds). Treat it as time axis, not an ECG lead.
    for k in ("t", "time", "time_s", "time_ms", "sec", "secs", "second", "seconds"):
        if k in cols_l:
            time_idx = cols_l[k]
            break

    leads: dict[str, np.ndarray] = {}
    for name, i in cols.items():
        if time_idx is not None and i == time_idx:
            continue
        if name == "":
            continue
        leads[name] = arr[:, i].astype(np.float32)

    fs_guess = 500.0
    if time_idx is not None:
        t = arr[:, time_idx]
        if t.size >= 3:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                fs_guess = float(round(1.0 / dt))

    return leads, fs_guess


def save_qc_report_json(path: str, report: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

# =============================================================================================================
# Fi2 — qc_signal_from_leads (arrays) : stable + utile pour run_qc_on_csv.py Version 090226
# Updated 090226 — IEC gate (Fs/noise/drift), constant-lead check, filter warn dedup
# # UPDATED le 11/02/2026 12H26  le 13/02/2026 21H45
# IEC HARDENED le 15/02/2026 — iec_profile traceability + global warn dedup + strict gate clarity 1864 ligne
# =============================================================================================================

def qc_signal_from_leads(
    leads: Mapping[str, np.ndarray],
    fs_hz: float = 500.0,
    patient_sex: str = "U",
    *,
    iec_normalize: bool = True,
    notch_hz: Optional[float] = 50.0,
) -> Dict[str, Any]:
    """
    Fi2: QC "signal" depuis arrays de leads.
    Retour dict contract-ready shape:
      {status, reasons, warnings[{code,reason,(severity)}], metrics{...}}

    IEC Normalisation:
      - appliquée AVANT toute mesure si iec_normalize=True
      - warnings du filtre remontent dans warnings + reasons
      - IEC est appliqué au niveau des leads (pas uniquement sur x) pour éviter toute incohérence.

    Sampling strategy:
      - Keep fs_source (input) for traceability + source gating
      - Standardize internally to TARGET_FS_HZ (500 Hz) for QC computations (PRE only here)

    IEC HARDENED:
      - Add explicit iec_profile (traceability + compliance narrative)
      - Global warnings dedup by code at the end (not only IEC filter warnings)
      - Keep "status_strict" as product gate; "status_dataset_aligned" for dataset mapping
    """
    warnings: List[Dict[str, str]] = []
    reasons: List[str] = []
    status = "PASS"

    if not leads:
        _flag(warnings, reasons, "FAIL_NO_SIGNAL", "No leads found in CSV.", "FAIL")
        return {
            "status": "FAIL",
            "status_strict": "FAIL",
            "status_dataset_aligned": "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "warnings": warnings,
            "metrics": {},
        }

    # ---------------- normalize sex + fs early ----------------
    sex_in = str(patient_sex).upper().strip() if patient_sex is not None else "U"
    sex = _normalize_sex(sex_in if sex_in in ("M", "F", "U") else "U")

    # fs_source: sampling rate of incoming data (traceability + source gating)
    fs_source = float(fs_hz) if _is_finite(fs_hz) and float(fs_hz) > 0 else float(TARGET_FS_HZ)

    # We'll build metrics progressively
    metrics: Dict[str, Any] = {
        # Backward compatibility: fs_hz will become internal fs after resampling
        "fs_hz": None,
        "fs_source_hz": float(fs_source),
        "iec_band": "0.05-150Hz",
        "sampling": {},
        # IEC hardened: explicit compliance profile (filled progressively)
        "iec_profile": {
            "reference": "IEC 60601-2-25 (signal acquisition & diagnostic ECG requirements)",
            "fs_source_hz": float(fs_source),
            "fs_min_hz": 250.0,
            "fs_recommended_hz": 500.0,
            "fs_internal_hz": None,          # set after resample
            "bandpass_hz": [0.05, 150.0],
            "notch_hz": None,                # set after validation
            "normalize_requested": bool(iec_normalize),
            "normalize_applied": False,      # set after _iec_normalize_leads
            "notes": [
                "Fs<250Hz => FAIL (IEC minimum). 250-499Hz => WARN (recommended 500Hz).",
                "Noise RMS target <30µV (WARN >30µV, FAIL >50µV).",
                "Baseline wander gate based on robust drift estimate (WARN >=0.60mV, FAIL >=2.00mV).",
            ],
        },
    }

    # ---------------- IEC gate: Fs (SOURCE) ----------------
    # IEC/AHA: Fs >= 250 Hz minimum; 500 Hz commonly used for diagnostic precision.
    if fs_source < 250.0:
        _flag(warnings, reasons, "FAIL_FS_TOO_LOW", f"Sampling rate too low ({fs_source:.1f} Hz).", "FAIL")
        return {
            "status": "FAIL",
            "status_strict": "FAIL",
            "status_dataset_aligned": "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "warnings": warnings,
            "metrics": metrics,
        }
    elif fs_source < 500.0:
        _flag(
            warnings,
            reasons,
            "WARN_FS_BELOW_RECOMMENDED",
            f"Sampling rate below recommended 500 Hz ({fs_source:.1f} Hz). Internal standardization to {float(TARGET_FS_HZ):.0f} Hz will be applied.",
            "WARN",
        )
        status = _worse_status(status, "WARN")

    # --- Baseline drift estimator (shared; used for per-lead raw/post-IEC drift maps) ---
    def _drift_mv_one_lead(x: np.ndarray, fs_local: float) -> Optional[float]:
        """
        Robust baseline drift estimator (mV) for a single lead.
        Uses a slow baseline proxy (moving average) then p95-p5 peak-to-peak.
        """
        try:
            x = np.asarray(x, dtype=np.float32)
            if x.ndim != 1:
                return None
            if x.size < int(1.0 * fs_local):  # need >=1s
                return None
            if not np.isfinite(x).any():
                return None

            win = int(max(1, round(0.8 * fs_local)))  # ~0.8s window
            if win >= x.size:
                return None

            k = np.ones((win,), dtype=np.float32) / float(win)
            baseline = np.convolve(x, k, mode="same")

            p5 = float(np.nanpercentile(baseline, 5))
            p95 = float(np.nanpercentile(baseline, 95))
            d = p95 - p5
            if not np.isfinite(d):
                return None
            return float(d)
        except Exception:
            return None

    # --- RAW baseline drift per-lead (pre-IEC) computed on SOURCE ---
    raw_baseline_drift_mv_by_lead: Dict[str, float] = {}
    for lname, sig in (leads or {}).items():
        d = _drift_mv_one_lead(sig, float(fs_source))
        if d is not None and np.isfinite(d):
            raw_baseline_drift_mv_by_lead[str(lname)] = float(d)
    raw_baseline_drift_mv = max(raw_baseline_drift_mv_by_lead.values()) if raw_baseline_drift_mv_by_lead else None

    # --- notch_hz validation (must be independent of iec_normalize) ---
    notch_invalid = False
    try:
        if notch_hz is None:
            notch_invalid = False
        else:
            nhz = float(notch_hz)
            if (not np.isfinite(nhz)) or (nhz not in (50.0, 60.0)):
                notch_invalid = True
    except Exception:
        notch_invalid = True

    if notch_invalid:
        _flag(
            warnings,
            reasons,
            "WARN_IEC_NOTCH_INVALID",
            f"Invalid notch_hz={notch_hz}. Expected 50 or 60 Hz. Notch disabled.",
            "WARN",
        )
        status = _worse_status(status, "WARN")
        notch_hz = None

    # IEC hardened: record notch decision early
    metrics["iec_profile"]["notch_hz"] = float(notch_hz) if (notch_hz is not None and _is_finite(notch_hz)) else None

    # ============================================================================
    # Internal standardization to TARGET_FS_HZ (PRE only)
    # ============================================================================
    try:
        leads_raw_500, rs_contract, rs_meta = resample_leads_to_500hz(leads, fs_source, prefer_polyphase=True)
    except Exception as e:
        _flag(warnings, reasons, "FAIL_RESAMPLE_INTERNAL", f"Internal resampling to {float(TARGET_FS_HZ):.0f} Hz failed: {e}", "FAIL")
        return {
            "status": "FAIL",
            "status_strict": "FAIL",
            "status_dataset_aligned": "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "warnings": warnings,
            "metrics": metrics,
        }

    fs_internal = float(getattr(rs_contract, "fs_internal_hz", TARGET_FS_HZ)) if rs_contract is not None else float(TARGET_FS_HZ)
    if not _is_finite(fs_internal) or fs_internal <= 0:
        fs_internal = float(TARGET_FS_HZ)

    # Attach sampling contract to metrics
    metrics["sampling"] = {
        "fs_source_hz": getattr(rs_contract, "fs_source_hz", None),
        "fs_internal_hz": float(fs_internal),
        "target_fs_hz": float(TARGET_FS_HZ),
        "resample_applied": bool(getattr(rs_contract, "resample_applied", False)),
        "resample_method": str(getattr(rs_contract, "resample_method", "unknown")),
        "resample_ratio": getattr(rs_contract, "resample_ratio", None),
        "fs_low_warning": bool(getattr(rs_contract, "fs_low_warning", False)),
        "notes": getattr(rs_contract, "notes", None),
        "per_lead": (rs_meta or {}).get("per_lead", {}),
    }

    # Backward compatibility: fs_hz now reflects INTERNAL fs used for QC measures
    metrics["fs_hz"] = float(fs_internal)

    # IEC hardened: record internal fs
    metrics["iec_profile"]["fs_internal_hz"] = float(fs_internal)

    # Basic sanity: resample must output something usable
    if not leads_raw_500:
        _flag(warnings, reasons, "FAIL_NO_SIGNAL", "No usable leads after internal resampling.", "FAIL")
        return {
            "status": "FAIL",
            "status_strict": "FAIL",
            "status_dataset_aligned": "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "warnings": warnings,
            "metrics": metrics,
        }

    # -------- IEC normalisation (bandpass + optional notch) BEFORE MEASURES --------
    leads_proc: Mapping[str, np.ndarray] = leads_raw_500
    iec_applied = False
    iec_filter_metrics: Dict[str, Any] = {}

    if iec_normalize:
        filtered, w, f_metrics = _iec_normalize_leads(leads_raw_500, fs_internal, notch_hz=notch_hz)

        if isinstance(filtered, dict) and len(filtered) > 0:
            leads_proc = filtered
            iec_applied = True

        # propagate IEC warnings (DEDUP by code locally)
        seen_codes = set()
        if isinstance(w, list):
            for ww in w:
                if isinstance(ww, dict):
                    code = ww.get("code", "WARN_IEC_FILTER")
                    if code in seen_codes:
                        continue
                    seen_codes.add(code)
                    _flag(
                        warnings,
                        reasons,
                        code,
                        ww.get("reason", "IEC filter warning."),
                        ww.get("severity", "WARN"),
                    )
                else:
                    code = "WARN_IEC_FILTER_NOTE"
                    if code not in seen_codes:
                        seen_codes.add(code)
                        _flag(warnings, reasons, code, str(ww), "WARN")
        elif w is not None:
            _flag(warnings, reasons, "WARN_IEC_FILTER_NOTE", str(w), "WARN")

        iec_filter_metrics = f_metrics if isinstance(f_metrics, dict) else {}

    # IEC hardened: record normalization actual
    metrics["iec_profile"]["normalize_applied"] = bool(iec_applied)

    # --- POST-IEC baseline drift per-lead (for reporting) in INTERNAL fs ---
    baseline_drift_mv_by_lead: Dict[str, float] = {}
    for lname, sig in (leads_proc or {}).items():
        d = _drift_mv_one_lead(sig, float(fs_internal))
        if d is not None and np.isfinite(d):
            baseline_drift_mv_by_lead[str(lname)] = float(d)
    baseline_drift_mv = max(baseline_drift_mv_by_lead.values()) if baseline_drift_mv_by_lead else None

    # ---------------------------------------------------------------------------
    # Pick representative lead for QC metrics (INTERNAL fs)
    x = None
    pick = None
    for k in ("II", "I", "V5", "V2", "V1", "III"):
        if k in leads_proc and leads_proc[k] is not None and len(leads_proc[k]) > 10:
            x = np.asarray(leads_proc[k], dtype=np.float32)
            pick = k
            break
    if x is None:
        for k, v in leads_proc.items():
            if v is not None and len(v) > 10:
                x = np.asarray(v, dtype=np.float32)
                pick = k
                break
    if x is None:
        _flag(warnings, reasons, "FAIL_NO_SIGNAL", "No usable lead arrays found.", "FAIL")
        return {
            "status": "FAIL",
            "status_strict": "FAIL",
            "status_dataset_aligned": "FAIL",
            "reasons": list(dict.fromkeys(reasons)),
            "warnings": warnings,
            "metrics": metrics,
        }

    # Prefer RAW (pre-IEC) pick lead for dropout checks, but still in INTERNAL fs space
    x_raw = None
    try:
        if pick is not None and pick in leads_raw_500 and leads_raw_500[pick] is not None and len(leads_raw_500[pick]) > 10:
            x_raw = np.asarray(leads_raw_500[pick], dtype=np.float32)
    except Exception:
        x_raw = None

    # ---------------- Constant/zero lead detection across leads (SOURCE) ----------------
    def _lead_is_constant(y: np.ndarray, fs_local: float, eps_std: float = 1e-6, eps0: float = 1e-4) -> bool:
        y = np.asarray(y, dtype=np.float32)
        if y.size < int(fs_local * 0.2):
            return False
        if not np.isfinite(y).all():
            return True
        if float(np.std(y)) < eps_std:
            return True
        frac_zero = float((np.abs(y) < eps0).mean())
        return frac_zero > 0.98

    bad: List[str] = []
    lead_stats: Dict[str, Any] = {}
    if isinstance(leads, dict):
        for k, v in leads.items():
            if v is None:
                continue
            yy = np.asarray(v, dtype=np.float32)
            if yy.size < 10:
                continue
            frac_zero = float((np.abs(yy) < 1e-4).mean())
            stdv = float(np.std(yy)) if yy.size > 1 else 0.0
            lead_stats[str(k)] = {"n": int(yy.size), "std": stdv, "frac_zero": frac_zero}
            if _lead_is_constant(yy, fs_source):
                bad.append(str(k))

    metrics["lead_stats"] = lead_stats
    metrics["constant_leads"] = bad
    if len(bad) >= 1:
        _flag(warnings, reasons, "FAIL_SIGNAL_CONSTANT_LEAD", f"Constant/zero leads detected: {bad}", "FAIL")
        status = "FAIL"

    # ---------------- Dropout / flat segment detection (INTERNAL timing) ----------------
    x_for_drop = x_raw if x_raw is not None else x
    x0 = np.asarray(x_for_drop, dtype=np.float32)

    if x0.size >= 3:
        eps_dx = 1e-5
        dx = np.abs(np.diff(x0))
        is_flat = dx < eps_dx

        max_run = 0
        run = 0
        for b in is_flat:
            if bool(b):
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 0

        max_run_ms = (float(max_run + 1) * 1000.0) / fs_internal
        frac_flat = float(is_flat.mean())

        eps0 = 1e-4
        frac_zero = float((np.abs(x0) < eps0).mean())

        metrics["dropout"] = {
            "lead_for_drop": str(pick),
            "used_raw": bool(x_raw is not None),
            "eps_dx": float(eps_dx),
            "max_flat_run_ms": float(max_run_ms),
            "frac_flat_diffs": float(frac_flat),
            "eps0": float(eps0),
            "frac_zero": float(frac_zero),
        }

        if max_run_ms >= 300.0:
            _flag(warnings, reasons, "FAIL_SIGNAL_DROPOUT", f"Detected flat/dropout segment ~{max_run_ms:.0f} ms.", "FAIL")
            status = "FAIL"
        else:
            # IEC HARDENED: reduce false positives on clean signals
            # - flat_diffs alone can be high on quantized/plateau segments
            # - require stronger evidence (long run, high near-zero, or extreme flatness)
            warn_flat_diffs = (frac_flat >= 0.60)
            warn_long_run  = (max_run_ms >= 250.0)
            warn_near_zero = (frac_zero >= 0.20)

            if warn_long_run or warn_near_zero or warn_flat_diffs:
                _flag(
                    warnings,
                    reasons,
                    "WARN_SIGNAL_DROPOUT",
                    f"Suspicious flatness (flat_diffs={frac_flat:.2%}, near_zero={frac_zero:.2%}, max_run_ms={max_run_ms:.0f}).",
                    "WARN",
                )
                status = _worse_status(status, "WARN")


    # ---------------- SNR / baseline drift / noise RMS (INTERNAL fs) ----------------
    noise_rms_uv = None
    baseline_drift_mv_pick = None
    filter_strength = 0.0

    try:
        out = _robust_snr_and_baseline(x, fs_internal)
        if isinstance(out, tuple) and len(out) == 4:
            snr_med, baseline_drift_mv_pick, filter_strength, noise_rms_uv = out
        else:
            snr_med, baseline_drift_mv_pick, filter_strength = out
    except Exception:
        snr_med, baseline_drift_mv_pick, filter_strength = 0.0, None, 0.0
        noise_rms_uv = None

    if noise_rms_uv is not None and _is_finite(noise_rms_uv):
        metrics["noise_rms_uv"] = float(noise_rms_uv)
        if float(noise_rms_uv) > 50.0:
            _flag(warnings, reasons, "FAIL_NOISE_RMS_HIGH", f"Noise RMS too high ({float(noise_rms_uv):.1f} µV).", "FAIL")
            status = "FAIL"
        elif float(noise_rms_uv) > 30.0:
            _flag(warnings, reasons, "WARN_NOISE_RMS_HIGH", f"Noise RMS elevated ({float(noise_rms_uv):.1f} µV).", "WARN")
            status = _worse_status(status, "WARN")
    else:
        metrics["noise_rms_uv"] = None

    raw_baseline_drift_mv_pick = None
    if x_raw is not None:
        try:
            out_raw = _robust_snr_and_baseline(x_raw, fs_internal)
            if isinstance(out_raw, tuple) and len(out_raw) >= 2:
                raw_baseline_drift_mv_pick = float(out_raw[1])
        except Exception:
            raw_baseline_drift_mv_pick = None

    # RR/HR (INTERNAL)
    peaks = _detect_r_peaks(x, fs_internal)
    rr_ms = None
    hr_bpm = None
    if getattr(peaks, "size", 0) >= 2:
        rr_ms = float(np.median(np.diff(peaks)) * 1000.0 / fs_internal)
        if rr_ms > 0:
            hr_bpm = float(60000.0 / rr_ms)

    # intervals (INTERNAL)
    qrs_ms_est, qt_ms_est, pr_ms_est = _estimate_intervals_ms(x, fs_internal)

    # qtc
    qtc_ms_est = None
    if _is_finite(qt_ms_est) and _is_finite(rr_ms):
        qtc_ms_est = _calc_qtc_bazett(float(qt_ms_est), float(rr_ms))

    # limb consistency + swap (INTERNAL, post-IEC)
    limb_consistency_err = _limb_consistency_error(leads_proc, fs_internal)
    best_hyp, best_err, normal_err = _infer_limb_swap(leads_proc, fs_internal)

    # --- SNR gates (product-safe strict policy) --- Updated Feb 16th, 2026. After "yes" for "“QC gate” = sécurité IA / démo médicale responsable."
    # Decision: FAIL if SNR < 4.0 ; WARN if 4.0 <= SNR < 5.0
    try:
        snr_val = float(snr_med)
    except Exception:
        snr_val = 0.0

    if snr_val < 4.0:
        _flag(warnings, reasons, "FAIL_LOW_SNR", f"SNR too low ({snr_val:.1f}).", "FAIL")
        status = "FAIL"
    elif snr_val < 5.0:
        _flag(warnings, reasons, "WARN_LOW_SNR", f"SNR marginal ({snr_val:.1f}).", "WARN")
        status = _worse_status(status, "WARN")

    # --- baseline drift gating ---
    BW_WARN_MV = 0.60
    BW_FAIL_MV = 2.00
    drift_for_gate = raw_baseline_drift_mv if _is_finite(raw_baseline_drift_mv) else baseline_drift_mv
    if _is_finite(drift_for_gate):
        d = float(drift_for_gate)
        if d >= BW_FAIL_MV:
            _flag(warnings, reasons, "FAIL_BASELINE_DRIFT", f"Baseline drift too high ({d:.2f} mV).", "FAIL")
            status = "FAIL"
        elif d >= BW_WARN_MV:
            _flag(warnings, reasons, "WARN_BASELINE_DRIFT", f"Baseline drift elevated ({d:.2f} mV).", "WARN")
            status = _worse_status(status, "WARN")

    if _is_finite(qrs_ms_est) and float(qrs_ms_est) >= QRS_WIDE:
        _flag(warnings, reasons, "WARN_QRS_WIDE", f"QRS wide ({float(qrs_ms_est):.0f} ms).", "WARN")
        status = _worse_status(status, "WARN")

    if _is_finite(qtc_ms_est) and sex in ("M", "F"):
        qtc = float(qtc_ms_est)
        if qtc >= QTc_HIGH_RISK:
            _flag(warnings, reasons, "WARN_QTC_HIGH_RISK", f"QTc ≥ {QTc_HIGH_RISK} ms ({qtc:.0f} ms).", "WARN")
            status = _worse_status(status, "WARN")
        elif qtc >= QTc_PROLONGED[sex]:
            _flag(warnings, reasons, "WARN_QTC_PROLONGED", f"QTc prolonged ({qtc:.0f} ms) for sex={sex}.", "WARN")
            status = _worse_status(status, "WARN")
        elif qtc <= QTc_SHORT:
            _flag(warnings, reasons, "WARN_QTC_SHORT", f"QTc short ({qtc:.0f} ms).", "WARN")
            status = _worse_status(status, "WARN")

    # limb swap hypothesis: only warn if it improves materially
    limb_swap_hypothesis = None
    if best_hyp is not None and best_err is not None and normal_err is not None:
        limb_swap_hypothesis = best_hyp
        if best_hyp != "NORMAL":
            if (best_err < 0.15) and (best_err <= 0.7 * normal_err):
                _flag(
                    warnings,
                    reasons,
                    f"WARN_LIMB_SWAP_{best_hyp}",
                    f"Possible limb lead swap hypothesis={best_hyp} (err {best_err:.3f} vs normal {normal_err:.3f}).",
                    "WARN",
                )
                status = _worse_status(status, "WARN")

    # ensure FAIL if any FAIL_*
    if any(str(w.get("code", "")).startswith("FAIL_") for w in warnings if isinstance(w, dict)):
        status = "FAIL"

    # If any WARN present and status still PASS -> bump to WARN
    if status == "PASS":
        for w in warnings:
            code = str(w.get("code", ""))
            sev = str(w.get("severity", ""))
            if sev == "WARN" or code.startswith("WARN_"):
                status = "WARN"
                break

    metrics.update(
        {
            "hr_bpm": hr_bpm,
            "rr_ms": rr_ms,
            "pr_ms_est": pr_ms_est,
            "qrs_ms_est": qrs_ms_est,
            "qt_ms_est": qt_ms_est,
            "qtc_ms_est": qtc_ms_est,
            "snr_med": float(snr_med) if _is_finite(snr_med) else None,
            "baseline_drift_mv": float(baseline_drift_mv) if _is_finite(baseline_drift_mv) else None,
            "raw_baseline_drift_mv": float(raw_baseline_drift_mv) if _is_finite(raw_baseline_drift_mv) else None,
            "baseline_drift_mv_by_lead": baseline_drift_mv_by_lead if "baseline_drift_mv_by_lead" in locals() else None,
            "raw_baseline_drift_mv_by_lead": raw_baseline_drift_mv_by_lead if "raw_baseline_drift_mv_by_lead" in locals() else None,
            "baseline_drift_mv_pick": float(baseline_drift_mv_pick) if _is_finite(baseline_drift_mv_pick) else None,
            "raw_baseline_drift_mv_pick": float(raw_baseline_drift_mv_pick) if _is_finite(raw_baseline_drift_mv_pick) else None,
            "filter_strength": float(filter_strength) if _is_finite(filter_strength) else None,
            "limb_consistency_err": float(limb_consistency_err) if limb_consistency_err is not None else None,
            "limb_swap_hypothesis": limb_swap_hypothesis,
            "lead_used": pick,
            "iec_normalized": bool(iec_applied),
            "iec_notch_hz": float(notch_hz) if (notch_hz is not None and _is_finite(notch_hz)) else None,
            "iec_filter": iec_filter_metrics,
        }
    )

    # --- Limb electrode reversal suspicion (Option B) ---
    try:
        rev = detect_limb_reversal(leads)
        metrics["limb_swap_kind"]  = rev["kind"]
        metrics["limb_swap_score"] = rev["score_best"]
        metrics["limb_swap_delta"] = rev["delta"]

        if rev["kind"] != "NONE":
            # Conservative gating (avoid false positives)
            if rev["score_best"] >= 0.60 and rev["delta"] >= 0.25:
                status_strict = "FAIL"
                reasons.append(f"Suspected limb electrode reversal ({rev['kind']}) high-confidence.")
                warnings.append({
                    "code": "FAIL_LIMB_LEAD_REVERSAL_SUSPECTED",
                    "reason": f"Limb electrode reversal suspected: {rev['kind']} (score={rev['score_best']:.2f}, delta={rev['delta']:.2f}).",
                    "severity": "high"
                })
            elif rev["score_best"] >= 0.45 and rev["delta"] >= 0.15:
                warnings.append({
                    "code": "WARN_LIMB_LEAD_REVERSAL_POSSIBLE",
                    "reason": f"Possible limb lead reversal: {rev['kind']} (score={rev['score_best']:.2f}, delta={rev['delta']:.2f}).",
                    "severity": "medium"
                })
    except Exception:
        # keep QC robust: never crash on limb reversal heuristic
        pass

    # ----------------------------------------------------------------------
    # IEC HARDENED: Global dedup warnings by "code" (keeps first occurrence)
    # ----------------------------------------------------------------------
    deduped: List[Dict[str, str]] = []
    seen = set()
    for w in warnings:
        if not isinstance(w, dict):
            continue
        code = str(w.get("code", "")).strip()
        if not code:
            code = "WARN_UNSPECIFIED"
            w["code"] = code
        if code in seen:
            continue
        seen.add(code)
        # normalize severity key (some callers may omit)
        if "severity" not in w or not w.get("severity"):
            if code.startswith("FAIL_"):
                w["severity"] = "FAIL"
            elif code.startswith("WARN_"):
                w["severity"] = "WARN"
            else:
                w["severity"] = "WARN"
        deduped.append(w)
    warnings = deduped

    # reasons: dedup + stable order
    reasons = list(dict.fromkeys(reasons))

    # ----------------------------------------------------------------------
    # Compute strict QC gate (signal quality only) — ignores clinical WARN_*
    # IMPORTANT: compute_strict() must be defined near CLINICAL_WARN_CODES
    # ----------------------------------------------------------------------
    status_strict = compute_strict(status, warnings)

    dataset_payload = {
        "status": status,               # dataset-oriented status (unchanged logic)
        "status_strict": status_strict, # available for mapping if needed
        "metrics": metrics,
        "warnings": warnings,
        "reasons": reasons,
    }
    dataset_status = _map_to_dataset_status(dataset_payload)

    return {
        "status": status_strict,        # product gate uses strict
        "status_strict": status_strict,
        "status_dataset_aligned": dataset_status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": metrics,
    }


# Inserption Feb 12th, 2026 20H30

def qc_signal_post_from_leads(
    leads_pred: Mapping[str, np.ndarray],
    fs_internal_hz: float = 500.0,
    *,
    expected_leads: Optional[List[str]] = None,
    amp_abs_fail_mv: float = 20.0,
    amp_abs_warn_mv: float = 10.0,
    constant_std_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    QC POST (post-inference): validate MedGemma-generated signals (or any model output).
    Contract shape:
      {status, reasons, warnings[{code,reason,severity}], metrics{...}}

    POST philosophy:
      - fast + defensive: catch degenerate outputs (NaNs/Infs, constants, absurd amplitude, missing leads)
      - assumes sampling already standardized (default 500 Hz)
    """
    warnings: List[Dict[str, Any]] = []
    reasons: List[str] = []
    status = "PASS"

    if expected_leads is None:
        expected_leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    fs_internal = float(fs_internal_hz) if np.isfinite(fs_internal_hz) and float(fs_internal_hz) > 0 else 500.0

    metrics: Dict[str, Any] = {
        "fs_hz": float(fs_internal),
        "n_leads_in": int(len(leads_pred) if leads_pred else 0),
        "lead_stats": {},
        "missing_leads": [],
        "post_checks": {
            "amp_abs_warn_mv": float(amp_abs_warn_mv),
            "amp_abs_fail_mv": float(amp_abs_fail_mv),
            "constant_std_eps": float(constant_std_eps),
        },
    }

    if not leads_pred:
        _flag(warnings, reasons, "FAIL_POST_NO_LEADS", "No predicted leads provided (post-inference).", "FAIL")
        return {"status": "FAIL", "reasons": list(dict.fromkeys(reasons)), "warnings": warnings, "metrics": metrics}

    # Missing leads (WARN by default)
    missing = [k for k in expected_leads if k not in leads_pred]
    if missing:
        metrics["missing_leads"] = missing
        _flag(
            warnings,
            reasons,
            "WARN_POST_MISSING_LEADS",
            f"Missing expected leads in post-inference output: {missing}",
            "WARN",
        )
        status = _worse_status(status, "WARN")

    # Per-lead checks
    bad_const = []
    bad_nonfinite = []
    bad_amp_fail = []
    bad_amp_warn = []

    for k, v in (leads_pred or {}).items():
        if v is None:
            continue
        x = np.asarray(v, dtype=np.float32).reshape(-1)

        # stats
        n = int(x.size)
        if n == 0:
            continue

        finite_mask = np.isfinite(x)
        frac_nonfinite = float(1.0 - float(finite_mask.mean()))
        x_f = x[finite_mask] if finite_mask.any() else np.asarray([], dtype=np.float32)

        stdv = float(np.std(x_f)) if x_f.size > 1 else 0.0
        maxabs = float(np.max(np.abs(x_f))) if x_f.size > 0 else float("inf")

        metrics["lead_stats"][str(k)] = {
            "n": n,
            "std": stdv,
            "max_abs_mv": maxabs,
            "frac_nonfinite": frac_nonfinite,
        }

        # non-finite -> FAIL
        if frac_nonfinite > 0.0:
            bad_nonfinite.append(str(k))

        # constant/near-constant -> FAIL
        if stdv < float(constant_std_eps):
            bad_const.append(str(k))

        # amplitude sanity
        if maxabs >= float(amp_abs_fail_mv):
            bad_amp_fail.append(str(k))
        elif maxabs >= float(amp_abs_warn_mv):
            bad_amp_warn.append(str(k))

    if bad_nonfinite:
        _flag(
            warnings,
            reasons,
            "FAIL_POST_NONFINITE",
            f"Non-finite values (NaN/Inf) detected in leads: {bad_nonfinite}",
            "FAIL",
        )
        status = "FAIL"

    if bad_const:
        _flag(
            warnings,
            reasons,
            "FAIL_POST_CONSTANT",
            f"Constant/near-constant predicted leads detected: {bad_const}",
            "FAIL",
        )
        status = "FAIL"

    if bad_amp_fail:
        _flag(
            warnings,
            reasons,
            "FAIL_POST_AMP_ABS",
            f"Abs amplitude too high (>= {amp_abs_fail_mv} mV) in leads: {bad_amp_fail}",
            "FAIL",
        )
        status = "FAIL"
    elif bad_amp_warn:
        _flag(
            warnings,
            reasons,
            "WARN_POST_AMP_ABS",
            f"High abs amplitude (>= {amp_abs_warn_mv} mV) in leads: {bad_amp_warn}",
            "WARN",
        )
        status = _worse_status(status, "WARN")

    # ensure FAIL if any FAIL_*
    if any(str(w.get("code", "")).startswith("FAIL_") for w in warnings if isinstance(w, dict)):
        status = "FAIL"

    return {
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "warnings": warnings,
        "metrics": metrics,
    }

### --- map to dataset status (covers WARN and FAIL forms) ---- ###  Created Feb 10th, 2026 Updated Feb 11th, 13th, 2026

def _map_to_dataset_status(qc_out: Dict[str, Any]) -> str:
    """
    Dataset-aligned status mapping (pedagogical demo alignment).
    Does NOT replace strict IEC status.

    Updated:
      - Takes qc_out["warnings"] codes into account (not just metrics),
        especially:
          * WARN_IEC_NOTCH_INVALID (and FAIL_IEC_NOTCH_INVALID if ever used)
          * WARN_BASELINE_DRIFT / FAIL_BASELINE_DRIFT (both forms)
      - Aligns drift thresholds with the dataset-aligned gating:
          WARN if drift >= 0.60 mV
          FAIL if drift >= 2.00 mV (extreme)
    """
    m = qc_out.get("metrics", {}) or {}
    warnings = qc_out.get("warnings", []) or []

    # ---- Collect codes (robust / normalized) ----
    codes = set()
    notch_hint = False  # fallback if warning is not a dict / code missing

    for w in warnings:
        if isinstance(w, dict):
            c_raw = w.get("code", "")
            c = str(c_raw or "").strip().upper()
            if c:
                codes.add(c)

            # fallback: sometimes code exists but is weird / empty -> look at reason
            r = str(w.get("reason", "") or "")
            if "NOTCH" in r.upper() and "INVALID" in r.upper():
                notch_hint = True

        else:
            # warning may be a string or something else
            s = str(w)
            su = s.upper()
            if "NOTCH" in su and "INVALID" in su:
                notch_hint = True

    # ---- Notch invalid: dataset expects WARN (even if signal is otherwise clean) ----
    if ("WARN_IEC_NOTCH_INVALID" in codes) or ("FAIL_IEC_NOTCH_INVALID" in codes) or notch_hint:
        return "WARN"

    # ---- Baseline drift codes handling (covers WARN_ and FAIL_ forms) ----
    if "WARN_BASELINE_DRIFT" in codes:
        return "WARN"
    # FAIL_BASELINE_DRIFT is handled below with metric threshold, but if present and not extreme -> WARN
    strict_fail_drift = ("FAIL_BASELINE_DRIFT" in codes)

    # ---- Metrics (fallback / main policy) ----
    try:
        snr = float(m.get("snr_med", 0.0) or 0.0)
    except Exception:
        snr = 0.0

    # Prefer RAW drift if available; fallback to filtered
    drift = m.get("raw_baseline_drift_mv", None)
    if drift is None:
        drift = m.get("baseline_drift_mv", 0.0)
    try:
        drift = float(drift or 0.0)
    except Exception:
        drift = 0.0

    dropout = m.get("dropout", {}) or {}
    try:
        max_drop = float(dropout.get("max_flat_run_ms", 0.0) or 0.0)
    except Exception:
        max_drop = 0.0

    constant = m.get("constant_leads", []) or []

    # Dataset-aligned drift thresholds
    BW_WARN_MV = 0.60
    BW_FAIL_MV = 2.00

    # ---- FAIL rules ----
    if constant:
        return "FAIL"

    if max_drop >= 300.0:
        return "FAIL"

    if snr < 4.0:
        return "FAIL"

    # Baseline drift: FAIL only when extreme
    if drift >= BW_FAIL_MV:
        return "FAIL"

    # If strict emitted FAIL_BASELINE_DRIFT but metric isn't extreme, degrade to WARN
    if strict_fail_drift:
        return "WARN"

    # ---- WARN rules ----
    if snr < 10.0:
        return "WARN"

    if drift >= BW_WARN_MV:
        return "WARN"

    return "PASS"


# TERMINUS DU CODE QC.PY


