from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


# Canonical 12-lead list for Appli_MedGemma_ECG (do not change ordering lightly)
ALL_LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Standard 3x4 layout (short leads). Note: short II is present but will be overwritten by long II strip.
LEAD_LAYOUT = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]


def _interp_1d_to_len(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = int(n)
    if n <= 0:
        return np.zeros((0,), dtype=np.float32)
    if len(x) == n:
        return x.astype(np.float32, copy=False)
    if len(x) < 2:
        return np.zeros((n,), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.interp(x_new, x_old, x).astype(np.float32)


def digitize_map_to_series12(
    prob_map: np.ndarray,
    lead_nrows: dict,
    *,
    px_per_mm: Optional[float] = None,
    gain_mm_per_mV: float = 10.0,
    disable_trp: bool = True,
    use_mask: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Appli_MedGemma_ECG-ready digitizer (robust, deterministic).

    Key behaviors:
      - A/B polarity check per segment (seg vs 1-seg), deterministic.
      - Computes centerline via softmax over rows (stable even when map is near-flat).
      - Converts pixel displacement -> mV IF px_per_mm is provided and >0:
            mV = (delta_px / px_per_mm) / gain_mm_per_mV
        Else fallback to legacy normalized amplitude (relative units).
      - Skips short "II" segment in the 3x4 grid (it is overwritten by the long II strip).
      - No Kaggle/TRP shaping (norm/blend/sharpen) inside this function.
        (disable_trp is kept for API compatibility; actual shaping must be done outside and be auditable.)
      - Env override supported:
            TRP_POLARITY_MODE = AUTO | FORCE_A | FORCE_B (or A/B)
    Returns:
      dict {lead: np.ndarray float32}, for ALL_LEADS_12. Missing leads are filled with zeros of required length.
    """
    m = np.asarray(prob_map, dtype=np.float32)
    if m.ndim != 2:
        return {ld: np.zeros((int(lead_nrows.get(ld, 0)),), dtype=np.float32) for ld in ALL_LEADS_12}

    H, W = m.shape
    if H < 8 or W < 8:
        return {ld: np.zeros((int(lead_nrows.get(ld, 0)),), dtype=np.float32) for ld in ALL_LEADS_12}

    # Conservative crop margins (avoid border artifacts / annotations)
    top_margin  = int(np.clip(0.04 * H, 2, 20))
    bot_margin  = int(np.clip(0.04 * H, 2, 20))
    left_margin = int(np.clip(0.03 * W, 2, 20))
    right_margin= int(np.clip(0.03 * W, 2, 20))

    m = m[top_margin:H - bot_margin, left_margin:W - right_margin]
    H2, W2 = m.shape
    if H2 < 8 or W2 < 8:
        return {ld: np.zeros((int(lead_nrows.get(ld, 0)),), dtype=np.float32) for ld in ALL_LEADS_12}

    band_h = H2 // 4
    if band_h < 2:
        return {ld: np.zeros((int(lead_nrows.get(ld, 0)),), dtype=np.float32) for ld in ALL_LEADS_12}

    bands = [m[i * band_h:(i + 1) * band_h, :] for i in range(3)]
    band_long = m[3 * band_h:H2, :]  # keep remainder rows

    seg_w = W2 // 4
    if seg_w < 2:
        return {ld: np.zeros((int(lead_nrows.get(ld, 0)),), dtype=np.float32) for ld in ALL_LEADS_12}

    def _nan0(x: np.ndarray) -> np.ndarray:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _amp(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return 0.0
        return float(np.nanmax(a) - np.nanmin(a))

    def _nz(a: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return 0.0
        thr = 1e-6
        return float(np.mean(np.abs(a) > thr))

    def _centerline_y(pm: np.ndarray, *, repair: bool) -> np.ndarray:
        """
        Returns y-position (float32) for each column, in pixel row coordinates [0..h-1].
        Uses stable softmax weighting across rows.
        """
        pm = np.asarray(pm, dtype=np.float32)
        pm = _nan0(pm)
        h, w = pm.shape
        if h < 2 or w < 1:
            return np.zeros((max(w, 0),), dtype=np.float32)

        mn = float(np.min(pm))
        mx = float(np.max(pm))
        if (not np.isfinite(mn)) or (not np.isfinite(mx)) or (mx - mn) < 1e-12:
            return np.full((w,), (h - 1) * 0.5, dtype=np.float32)

        # Normalize to [0,1] locally (stable)
        p = (pm - mn) / (mx - mn + 1e-8)

        # Softmax over rows per column
        col_max = np.max(p, axis=0, keepdims=True)
        z = (p - col_max) * 25.0
        z = np.clip(z, -50.0, 50.0)
        wgt = np.exp(z).astype(np.float32, copy=False)

        yy = np.arange(h, dtype=np.float32).reshape(h, 1)
        denom = np.sum(wgt, axis=0) + 1e-8
        y = (np.sum(wgt * yy, axis=0) / denom).astype(np.float32)

        if repair:
            # bad columns only if almost flat and very low max
            col_max2 = np.max(p, axis=0).astype(np.float32)
            col_min2 = np.min(p, axis=0).astype(np.float32)
            col_rng = (col_max2 - col_min2)
            bad = (col_rng < 0.02) & (col_max2 < 0.10)
            if bad.any():
                x = np.arange(w, dtype=np.float32)
                good = ~bad
                if int(good.sum()) >= max(10, int(w * 0.05)):
                    y[bad] = np.interp(x[bad], x[good], y[good]).astype(np.float32)

        return y

    def _trace_from_seg(seg: np.ndarray) -> np.ndarray:
        """
        seg prob-map -> 1D signal.
        If px_per_mm provided -> mV, else normalized relative units.
        """
        seg = np.asarray(seg, dtype=np.float32)
        seg = _nan0(seg)
        h, w = seg.shape
        if h < 2 or w < 2:
            return np.zeros((max(w, 0),), dtype=np.float32)

        y = _centerline_y(seg, repair=bool(use_mask))
        base_px = float(np.median(y))
        amp_px = (base_px - y).astype(np.float32, copy=False)  # + upward

        if (px_per_mm is not None) and float(px_per_mm) > 0.0 and float(gain_mm_per_mV) > 0.0:
            amp_mm = (amp_px / float(px_per_mm)).astype(np.float32, copy=False)
            amp = (amp_mm / float(gain_mm_per_mV)).astype(np.float32, copy=False)
        else:
            scale = float(max(1.0, float(h)))
            amp = (amp_px / scale).astype(np.float32, copy=False)

        # Light smoothing
        if w >= 3:
            ker = np.ones((3,), dtype=np.float32) / 3.0
            amp = np.convolve(amp, ker, mode="same").astype(np.float32)

        return _nan0(amp)

    def _pick_pol(seg: np.ndarray) -> np.ndarray:
        """
        Choose between A=seg and B=1-seg using score = amp*nz.
        Override using env TRP_POLARITY_MODE: FORCE_A/FORCE_B/A/B.
        """
        sA = _trace_from_seg(seg)
        sB = _trace_from_seg(1.0 - seg)

        ampA, nzA = _amp(sA), _nz(sA)
        ampB, nzB = _amp(sB), _nz(sB)
        scA = ampA * nzA
        scB = ampB * nzB

        prefer = "B" if scB > scA else "A"
        pol_mode = os.getenv("TRP_POLARITY_MODE", "AUTO").strip().upper()
        if pol_mode in ("FORCE_A", "A"):
            prefer = "A"
        elif pol_mode in ("FORCE_B", "B"):
            prefer = "B"
        return sB if prefer == "B" else sA

    out: Dict[str, np.ndarray] = {}

    # 3x4 grid (short leads); skip short II (overwritten by long II later).
    for r in range(3):
        for c in range(4):
            lead = LEAD_LAYOUT[r][c]
            if lead == "II":
                continue
            seg = bands[r][:, c * seg_w:(c + 1) * seg_w]
            s = _pick_pol(seg)
            n = int(lead_nrows.get(lead, int(s.size)))
            out[lead] = _interp_1d_to_len(s, n)

    # Long lead II strip
    s_long = _pick_pol(band_long)
    n_long = int(lead_nrows.get("II", int(s_long.size)))
    out["II"] = _interp_1d_to_len(s_long, n_long)

    # Ensure all leads exist (zeros only; higher-level can repair/fill if needed).
    for lead in ALL_LEADS_12:
        if lead not in out:
            out[lead] = np.zeros((int(lead_nrows.get(lead, 0)),), dtype=np.float32)

    return out
