# qc.py  Version du 05.02.2026 ENRICHI DES BLOCS “QC Normes/Clinique/Mesures”

from __future__ import annotations

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
# 0) QC SIGNAL / "CLINICAL" (durci) — PASS/WARN/FAIL + flags code+reason
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

def iec_bandpass_filter(x: np.ndarray, fs: float, *, notch_hz: Optional[float] = None, order: int = 4):
    y, warnings = iec_bandpass_filter_1d(x, fs, notch_hz=notch_hz, order=order)
    return y, warnings


def iec_bandpass_filter_1d(
    x: np.ndarray,
    fs: float,
    *,
    notch_hz: Optional[float] = None,
    order: int = 4,
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    IEC-ish preprocessing for digital ECG (QC only):
      - bandpass 0.05–150 Hz (Butterworth, filtfilt)
      - optional notch (50/60Hz), Q=30
    Returns: (x_filt, warnings[])
    """
    warnings: List[Dict[str, str]] = []
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.copy(), warnings

    if not _is_finite(fs) or float(fs) <= 0:
        warnings.append({"code": "WARN_FILTER_BAD_FS", "reason": f"Invalid fs={fs}; IEC filter skipped"})
        return x.copy(), warnings

    try:
        from scipy import signal as sp_signal  # type: ignore
    except Exception:
        warnings.append({"code": "WARN_FILTER_SCIPY_MISSING", "reason": "scipy not available; IEC filter skipped"})
        return x.copy(), warnings

    fsf = float(fs)
    nyq = 0.5 * fsf

    lo = 0.05 / nyq
    hi = 150.0 / nyq

    # clamp if fs too low
    if hi >= 1.0:
        hi = min(0.99, hi)
        warnings.append({"code": "WARN_FILTER_FS_LOW_CLAMP", "reason": f"fs={fsf} too low for 150Hz; high cutoff clamped"})

    if lo <= 0 or lo >= 1.0 or hi <= 0 or hi >= 1.0 or lo >= hi:
        warnings.append({"code": "WARN_FILTER_BAD_BAND", "reason": f"Bad band params lo={lo:.4f}, hi={hi:.4f}; IEC filter skipped"})
        return x.copy(), warnings

    b, a = sp_signal.butter(int(order), [lo, hi], btype="band")
    xf = sp_signal.filtfilt(b, a, x).astype(np.float32)

    if notch_hz is not None and _is_finite(notch_hz) and float(notch_hz) > 0:
        w0 = float(notch_hz) / nyq
        if 0 < w0 < 1:
            bn, an = sp_signal.iirnotch(w0, Q=30.0)
            xf = sp_signal.filtfilt(bn, an, xf).astype(np.float32)
        else:
            warnings.append({"code": "WARN_FILTER_BAD_NOTCH", "reason": f"notch {notch_hz}Hz invalid for fs={fsf}"})

    return xf, warnings


def _iec_normalize_leads(
    leads: Mapping[str, np.ndarray],
    fs: float,
    *,
    notch_hz: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, str]]]:
    """
    Filtre IEC toutes les dérivations (recommandé pour cohérences inter-leads).
    Retour: (leads_filt, warnings)
    """
    warnings: List[Dict[str, str]] = []
    out: Dict[str, np.ndarray] = {}

    if not leads:
        return out, warnings

    any_ok = False
    for k, v in leads.items():
        if v is None:
            continue
        x = np.asarray(v, dtype=np.float32)
        if x.size == 0:
            continue
        xf, w = iec_bandpass_filter_1d(x, fs, notch_hz=notch_hz)
        out[k] = xf
        warnings.extend(w)
        any_ok = True

    if not any_ok:
        # fallback: return original as arrays
        for k, v in leads.items():
            if v is not None:
                out[k] = np.asarray(v, dtype=np.float32)
        warnings.append({"code": "WARN_FILTER_NO_USABLE_LEADS", "reason": "No usable leads to filter; using raw leads"})
    return out, warnings


def _robust_snr_and_baseline(x_mv: np.ndarray, fs: float) -> tuple[float, float, float]:
    """
    Returns (snr_med, baseline_drift_mv, filter_strength[0..1-ish]).
    Heuristics: no clinical guarantee; QC only.
    """
    x = np.asarray(x_mv, dtype=np.float32)
    if x.size < int(fs * 1.0):
        return 0.0, 0.0, 0.0

    win_base = int(max(3, round(fs * 0.8)))
    base = _moving_average(x, win_base)
    detr = x - base

    b5 = float(np.percentile(base, 5))
    b95 = float(np.percentile(base, 95))
    baseline_drift = float(b95 - b5)

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

    return snr, baseline_drift, filter_strength


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

    time_idx = None
    for k in ["t", "time", "sec", "seconds"]:
        if k in cols:
            time_idx = cols[k]
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


# =============================================================================
# Fi2 — qc_signal_from_leads (arrays) : stable + utile pour run_qc_on_csv.py
# =============================================================================

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
      {status, reasons, warnings[{code,reason}], metrics{...}}

    IEC Normalisation:
      - appliquée AVANT toute mesure si iec_normalize=True
      - warnings du filtre remontent dans warnings/reasons
    """
    warnings: List[Dict[str, str]] = []
    reasons: List[str] = []
    status = "PASS"

    if not leads:
        _flag(warnings, reasons, "FAIL_NO_SIGNAL", "No leads found in CSV.")
        return {"status": "FAIL", "reasons": reasons, "warnings": warnings, "metrics": {}}

    sex = _normalize_sex(patient_sex if patient_sex in ("M", "F") else ("F" if str(patient_sex).upper().startswith("F") else "M"))

    fs = float(fs_hz) if _is_finite(fs_hz) and float(fs_hz) > 0 else 500.0

    # -------- IEC normalisation (bandpass + optional notch) BEFORE MEASURES --------
    leads_proc: Mapping[str, np.ndarray] = leads
    iec_applied = False
    if iec_normalize:
        filtered, w = _iec_normalize_leads(leads, fs, notch_hz=notch_hz)
        if filtered:
            leads_proc = filtered
            iec_applied = True
        # push warnings into contract output (and reasons via qc_pack_v1)
        for ww in w:
            _flag(warnings, reasons, ww.get("code", "WARN_FILTER_UNKNOWN"), ww.get("reason", "IEC filter warning."))
    # ---------------------------------------------------------------------------

    # Pick a representative lead for timing/SNR metrics (from leads_proc)
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
        _flag(warnings, reasons, "FAIL_NO_SIGNAL", "No usable lead arrays found.")
        return {"status": "FAIL", "reasons": reasons, "warnings": warnings, "metrics": {}}

    snr_med, baseline_drift_mv, filter_strength = _robust_snr_and_baseline(x, fs)

    # RR/HR from R peaks
    peaks = _detect_r_peaks(x, fs)
    rr_ms = None
    hr_bpm = None
    if peaks.size >= 2:
        rr_ms = float(np.median(np.diff(peaks)) * 1000.0 / fs)
        if rr_ms > 0:
            hr_bpm = float(60000.0 / rr_ms)

    # intervals
    qrs_ms_est, qt_ms_est, pr_ms_est = _estimate_intervals_ms(x, fs)

    # qtc
    qtc_ms_est = None
    if _is_finite(qt_ms_est) and _is_finite(rr_ms):
        qtc_ms_est = _calc_qtc_bazett(float(qt_ms_est), float(rr_ms))

    # limb consistency + swap (use IEC-normalized leads if enabled)
    limb_consistency_err = _limb_consistency_error(leads_proc, fs)
    best_hyp, best_err, normal_err = _infer_limb_swap(leads_proc, fs)

    # heuristics → warnings/reasons/status
    if snr_med < 5.0:
        _flag(warnings, reasons, "FAIL_LOW_SNR", f"SNR too low ({snr_med:.1f}).")
        status = "FAIL"
    elif snr_med < 10.0:
        _flag(warnings, reasons, "WARN_LOW_SNR", f"SNR marginal ({snr_med:.1f}). Review recommended.")
        status = _worse_status(status, "WARN")

    if baseline_drift_mv > 0.5:
        _flag(warnings, reasons, "WARN_BASELINE_DRIFT", f"Baseline drift high ({baseline_drift_mv:.2f} mV).")
        status = _worse_status(status, "WARN")

    if _is_finite(qrs_ms_est) and float(qrs_ms_est) >= QRS_WIDE:
        _flag(warnings, reasons, "WARN_QRS_WIDE", f"QRS wide ({float(qrs_ms_est):.0f} ms).")
        status = _worse_status(status, "WARN")

    if _is_finite(qtc_ms_est):
        qtc = float(qtc_ms_est)
        if qtc >= QTc_HIGH_RISK:
            _flag(warnings, reasons, "WARN_QTC_HIGH_RISK", f"QTc ≥ {QTc_HIGH_RISK} ms ({qtc:.0f} ms).")
            status = _worse_status(status, "WARN")
        elif qtc >= QTc_PROLONGED[sex]:
            _flag(warnings, reasons, "WARN_QTC_PROLONGED", f"QTc prolonged ({qtc:.0f} ms) for sex={sex}.")
            status = _worse_status(status, "WARN")
        elif qtc <= QTc_SHORT:
            _flag(warnings, reasons, "WARN_QTC_SHORT", f"QTc short ({qtc:.0f} ms).")
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
                )
                status = _worse_status(status, "WARN")

    # ensure FAIL if any FAIL_*
    if any(str(w.get("code", "")).startswith("FAIL_") for w in warnings):
        status = "FAIL"

    metrics = {
        "hr_bpm": hr_bpm,
        "rr_ms": rr_ms,
        "pr_ms_est": pr_ms_est,
        "qrs_ms_est": qrs_ms_est,
        "qt_ms_est": qt_ms_est,
        "qtc_ms_est": qtc_ms_est,
        "snr_med": float(snr_med),
        "baseline_drift_mv": float(baseline_drift_mv),
        "filter_strength": float(filter_strength),
        "limb_consistency_err": float(limb_consistency_err) if limb_consistency_err is not None else None,
        "limb_swap_hypothesis": limb_swap_hypothesis,
        "fs_hz": float(fs),
        "lead_used": pick,
        "iec_normalized": bool(iec_applied),
        "iec_notch_hz": float(notch_hz) if (notch_hz is not None and _is_finite(notch_hz)) else None,
        "iec_band": "0.05-150Hz",
    }

    reasons = list(dict.fromkeys(reasons))
    return {"status": status, "reasons": reasons, "warnings": warnings, "metrics": metrics}


# =============================================================================
# Fi3 — qc_pack_v1 : compatible run_qc_on_csv.py (signal_qc=raw)
# =============================================================================

def qc_pack_v1(
    *,
    input_type: str,
    source_path: str,
    sample_id: Optional[str] = None,
    fs_hz: Optional[float] = None,
    signal_qc: Optional[Dict[str, Any]] = None,
    repo: str = "Appli_MedGem_PoC",
    commit: Optional[str] = None,
    qc_module: str = "medgem_poc.qc",
    qc_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fi3: empaquete en 'qc.v1' contract.
    - Si signal_qc est déjà un objet qc.v1 (schema=qc.v1), on le renvoie tel quel.
    - Sinon, on extrait {status, reasons, warnings, metrics} depuis signal_qc.
    """
    if isinstance(signal_qc, dict) and signal_qc.get("schema") == "qc.v1":
        return signal_qc

    qc_version = qc_version or QC_VERSION

    raw = signal_qc or {}
    status = raw.get("status", "WARN")
    reasons = raw.get("reasons", [])
    warnings = raw.get("warnings", [])
    metrics = raw.get("metrics", {})

    # normalize minimal types
    if not isinstance(reasons, list):
        reasons = []
    if not isinstance(warnings, list):
        warnings = []
    if not isinstance(metrics, dict):
        metrics = {}

    # ensure warnings codes are included in reasons
    rset = set([r for r in reasons if isinstance(r, str)])
    for w in warnings:
        if isinstance(w, dict) and isinstance(w.get("code"), str):
            if w["code"] not in rset:
                reasons.append(w["code"])
                rset.add(w["code"])

    return {
        "schema": "qc.v1",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input": {
            "type": input_type,
            "source_path": source_path,
            "sample_id": sample_id,
            "fs_hz": fs_hz,
        },
        "pipeline": {
            "repo": repo,
            "commit": commit,
            "qc_module": qc_module,
            "qc_version": qc_version,
        },
        "status": status,
        "reasons": reasons,
        "warnings": warnings,
        "metrics": metrics,
    }

# TERMINUS
