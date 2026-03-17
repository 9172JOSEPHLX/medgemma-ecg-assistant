# L:\Appli_MedGem_PoC\src\medgem_poc\input_adapters\ecg_digitization\convert.py MARS 16TH, 2026 V1.000

from __future__ import annotations

import os
import csv
import math
import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from scipy.signal import resample as fourier_resample  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from medgem_poc.vendor_ecg_poc_runtime import digitize_map_to_series12
from medgem_poc.input_adapters.ecg_digitization.regression_infer import RegressionInferConfig, infer_prob_map_from_path


LEADS12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


@dataclass
class ConvertConfig:
    fs_out: int = 500
    duration_s: float = 10.0
    gain_mm_per_mV: float = 10.0
    paper_speed_mm_per_s: float = 25.0

    # Conservative extraction QC thresholds (initial)
    fail_rect_conf: float = 0.55
    fail_calib_conf: float = 0.55
    fail_layout_conf: float = 0.60
    fail_cov_pct: float = 80.0
    fail_gap_ms: float = 200.0
    fail_clip_pct: float = 3.0
    fail_einth_p95_mV: float = 0.25
    fail_limb_p95_mV: float = 0.25

    warn_rect_low: float = 0.55
    warn_rect_high: float = 0.75
    warn_calib_low: float = 0.55
    warn_calib_high: float = 0.75
    warn_layout_low: float = 0.60
    warn_layout_high: float = 0.80
    warn_cov_low: float = 80.0
    warn_cov_high: float = 90.0
    warn_gap_low: float = 120.0
    warn_gap_high: float = 200.0
    warn_clip_low: float = 1.0
    warn_clip_high: float = 3.0
    warn_einth_low: float = 0.15
    warn_einth_high: float = 0.25
    warn_limb_low: float = 0.15
    warn_limb_high: float = 0.25


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _finite32(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _resample_to_len(x: np.ndarray, n: int) -> np.ndarray:
    x = _finite32(x)
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if x.size == n:
        return x
    if x.size < 2:
        return np.zeros((n,), dtype=np.float32)
    if _HAS_SCIPY:
        return np.asarray(fourier_resample(x, n), dtype=np.float32).reshape(-1)
    # fallback linear
    t0 = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    t1 = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(t1, t0, x).astype(np.float32)


def _coverage_and_gaps(x: np.ndarray, fs: int) -> Tuple[float, float]:
    a = _finite32(x)
    if a.size == 0:
        return 0.0, float("inf")
    nz = np.abs(a) > 1e-8
    cov = 100.0 * float(np.mean(nz))

    max_run = 0
    run = 0
    for v in (~nz):
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    gap_ms = 1000.0 * float(max_run) / float(max(1, int(fs)))
    return cov, gap_ms


def _clipping_pct(x: np.ndarray) -> float:
    a = _finite32(x)
    if a.size < 8:
        return 0.0
    lo = float(np.min(a))
    hi = float(np.max(a))
    if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-12:
        return 0.0
    eps = 1e-6 * max(1.0, abs(hi - lo))
    at = (np.abs(a - lo) <= eps) | (np.abs(a - hi) <= eps)
    return 100.0 * float(np.mean(at))


def _pstats_abs(x: np.ndarray) -> Dict[str, float]:
    a = np.abs(_finite32(x))
    if a.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
    }


def _ecg_physics_checks(leads: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    I   = _finite32(leads.get("I", np.zeros(0, np.float32)))
    II  = _finite32(leads.get("II", np.zeros(0, np.float32)))
    III = _finite32(leads.get("III", np.zeros(0, np.float32)))
    aVR = _finite32(leads.get("aVR", np.zeros(0, np.float32)))
    aVL = _finite32(leads.get("aVL", np.zeros(0, np.float32)))
    aVF = _finite32(leads.get("aVF", np.zeros(0, np.float32)))

    n = min(I.size, II.size, III.size)
    ein = (II[:n] - (I[:n] + III[:n])) if n > 0 else np.zeros(0, np.float32)

    n2 = min(aVR.size, aVL.size, aVF.size)
    limb = (aVR[:n2] + aVL[:n2] + aVF[:n2]) if n2 > 0 else np.zeros(0, np.float32)

    return {"einthoven": _pstats_abs(ein), "limb_sum": _pstats_abs(limb)}


def _classify_extract(metrics: Dict[str, Any], cfg: ConvertConfig) -> str:
    rect_c = float(metrics.get("rectification_confidence", 1.0))
    cal_c  = float(metrics.get("calibration_confidence", 1.0))
    lay_c  = float(metrics.get("layout_confidence", 1.0))
    cov    = float(metrics.get("trace_coverage_pct", 0.0))
    gap_ms = float(metrics.get("gap_max_ms", 1e9))
    clip   = float(metrics.get("clipping_pct", 0.0))
    ein_p95 = float(metrics.get("einthoven_residual_p95_mV", 0.0))
    lim_p95 = float(metrics.get("limb_sum_residual_p95_mV", 0.0))
    lead_ok = bool(metrics.get("lead_count", 0) == 12)

    if rect_c < cfg.fail_rect_conf or cal_c < cfg.fail_calib_conf or lay_c < cfg.fail_layout_conf:
        return "EXTRACT_FAIL"
    if not lead_ok:
        return "EXTRACT_FAIL"
    if cov < cfg.fail_cov_pct or gap_ms > cfg.fail_gap_ms or clip > cfg.fail_clip_pct:
        return "EXTRACT_FAIL"
    if ein_p95 > cfg.fail_einth_p95_mV or lim_p95 > cfg.fail_limb_p95_mV:
        return "EXTRACT_FAIL"

    warn = False
    if cfg.warn_rect_low <= rect_c < cfg.warn_rect_high: warn = True
    if cfg.warn_calib_low <= cal_c < cfg.warn_calib_high: warn = True
    if cfg.warn_layout_low <= lay_c < cfg.warn_layout_high: warn = True
    if cfg.warn_cov_low <= cov < cfg.warn_cov_high: warn = True
    if cfg.warn_gap_low < gap_ms <= cfg.warn_gap_high: warn = True
    if cfg.warn_clip_low < clip <= cfg.warn_clip_high: warn = True
    if cfg.warn_einth_low < ein_p95 <= cfg.warn_einth_high: warn = True
    if cfg.warn_limb_low < lim_p95 <= cfg.warn_limb_high: warn = True

    return "EXTRACT_WARN" if warn else "EXTRACT_PASS"


def _write_csv_wide(path: str, leads: Dict[str, np.ndarray], fs: int) -> None:
    n = min(int(leads[k].size) for k in LEADS12)
    time_s = (np.arange(n, dtype=np.float32) / float(fs)).astype(np.float32)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + LEADS12)
        for i in range(n):
            row = [f"{float(time_s[i]):.6f}"]
            for ld in LEADS12:
                row.append(f"{float(leads[ld][i]):.9g}")
            w.writerow(row)


def _write_manifest_minimal(path: str, manifest: Dict[str, Any]) -> None:
    import xml.etree.ElementTree as ET
    ns = "https://medgemma.ecg/manifest/v1"
    ET.register_namespace("mdg", ns)

    root = ET.Element(f"{{{ns}}}ecg_manifest", {"schema_version": "1.0"})

    def add(parent, tag, text=None, attrib=None):
        e = ET.SubElement(parent, f"{{{ns}}}{tag}", attrib or {})
        if text is not None:
            e.text = str(text)
        return e

    src = add(root, "source")
    for k in ("input_path","input_kind","input_sha256","page_index","render_dpi"):
        if k in manifest:
            add(src, k, manifest[k])

    exq = add(root, "extraction_qc", attrib={"status": str(manifest.get("extraction_status", ""))})
    for k in ("trace_coverage_pct","gap_max_ms","clipping_pct"):
        if k in manifest:
            add(exq, k, manifest[k])

    phy = add(root, "ecg_physics_checks")
    if "einthoven" in manifest:
        add(phy, "einthoven", attrib={
            "mean_mV": str(manifest["einthoven"].get("mean",0.0)),
            "median_mV": str(manifest["einthoven"].get("median",0.0)),
            "p95_mV": str(manifest["einthoven"].get("p95",0.0)),
        })
    if "limb_sum" in manifest:
        add(phy, "limb_sum", attrib={
            "mean_mV": str(manifest["limb_sum"].get("mean",0.0)),
            "median_mV": str(manifest["limb_sum"].get("median",0.0)),
            "p95_mV": str(manifest["limb_sum"].get("p95",0.0)),
        })

    arts = add(root, "artifacts")
    for item in manifest.get("artifacts", []):
        add(arts, item.get("tag","file"), attrib={"path": item.get("path",""), "sha256": item.get("sha256","")})

    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def convert_ecg_image(
    input_path: str,
    out_dir: str,
    *,
    cfg: Optional[ConvertConfig] = None,
    page: int = 0,
    dpi: int = 400,
    px_per_mm_mean: Optional[float] = None,
    rectification_confidence: float = 1.0,
    calibration_confidence: float = 1.0,
    layout_confidence: float = 1.0,
    prob_map_provider: Optional[Callable[[str], np.ndarray]] = None,
    regression_cfg: Optional[RegressionInferConfig] = None,
    prob_map_npy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Orchestrator (MVP):
      - If prob_map_npy is provided: loads prob_map from .npy
      - Else if prob_map_provider is provided: uses it
      - Else if regression_cfg is provided: runs torchscript regression infer
      - Else: raises

    Downstream:
      - digitize_map_to_series12 (vendor runtime)
      - resample to fs_out & duration
      - compute Extraction QC + ECG physics residuals
      - write CSV + minimal manifest XML v1
    """
    cfg = cfg or ConvertConfig()
    _ensure_dir(out_dir)

    if prob_map_npy:
        prob_map = np.asarray(np.load(prob_map_npy), dtype=np.float32)
    elif prob_map_provider is not None:
        prob_map = np.asarray(prob_map_provider(input_path), dtype=np.float32)
    elif regression_cfg is not None:
        prob_map = infer_prob_map_from_path(input_path, regression_cfg)
    else:
        raise RuntimeError("Need prob_map_npy OR prob_map_provider OR regression_cfg")

    if prob_map.ndim != 2:
        raise RuntimeError("prob_map must be 2D (H,W)")

    n_target = int(round(float(cfg.fs_out) * float(cfg.duration_s)))
    lead_nrows = {ld: n_target for ld in LEADS12}

    leads = digitize_map_to_series12(
        prob_map,
        lead_nrows,
        px_per_mm=px_per_mm_mean,
        gain_mm_per_mV=float(cfg.gain_mm_per_mV),
        disable_trp=True,
        use_mask=False,
    )

    for ld in LEADS12:
        leads[ld] = _resample_to_len(leads[ld], n_target)

    covs, gaps, clips = [], [], []
    for ld in LEADS12:
        c, g = _coverage_and_gaps(leads[ld], cfg.fs_out)
        covs.append(c); gaps.append(g); clips.append(_clipping_pct(leads[ld]))
    cov_min = float(np.min(covs)) if covs else 0.0
    gap_max = float(np.max(gaps)) if gaps else float("inf")
    clip_max = float(np.max(clips)) if clips else 0.0

    phy = _ecg_physics_checks(leads)

    metrics = {
        "lead_count": 12,
        "rectification_confidence": float(rectification_confidence),
        "calibration_confidence": float(calibration_confidence),
        "layout_confidence": float(layout_confidence),
        "trace_coverage_pct": float(cov_min),
        "gap_max_ms": float(gap_max),
        "clipping_pct": float(clip_max),
        "einthoven_residual_p95_mV": float(phy["einthoven"]["p95"]),
        "limb_sum_residual_p95_mV": float(phy["limb_sum"]["p95"]),
    }
    extraction_status = _classify_extract(metrics, cfg)

    csv_path = os.path.join(out_dir, "ecg_12lead_500hz.csv")
    _write_csv_wide(csv_path, leads, cfg.fs_out)
    csv_sha = _sha256_file(csv_path)

    man_path = os.path.join(out_dir, "manifest_v1.xml")
    in_sha = _sha256_file(input_path) if os.path.isfile(input_path) else ""
    units = "mV" if (px_per_mm_mean is not None and float(px_per_mm_mean) > 0.0) else "rel"

    manifest = {
        "input_path": input_path,
        "input_kind": os.path.splitext(input_path)[1].lstrip(".").upper(),
        "input_sha256": in_sha,
        "page_index": int(page),
        "render_dpi": int(dpi),
        "extraction_status": extraction_status,
        "trace_coverage_pct": float(cov_min),
        "gap_max_ms": float(gap_max),
        "clipping_pct": float(clip_max),
        "einthoven": phy["einthoven"],
        "limb_sum": phy["limb_sum"],
        "artifacts": [
            {"tag": "csv_wide", "path": os.path.basename(csv_path), "sha256": csv_sha},
        ],
        "signal_units": units,
    }
    _write_manifest_minimal(man_path, manifest)
    man_sha = _sha256_file(man_path)

    return {
        "extraction_status": extraction_status,
        "paths": {"csv": csv_path, "manifest": man_path},
        "sha256": {"csv": csv_sha, "manifest": man_sha},
        "metrics": metrics,
    }

# TERMINUS