# src/medgem_poc/qc/resample_to_500.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    # Preferred: polyphase FIR resampling (stable for ECG; includes anti-aliasing on downsample)
    from scipy.signal import resample_poly  # type: ignore
    _HAS_SCIPY = True
except Exception:
    resample_poly = None  # type: ignore
    _HAS_SCIPY = False


TARGET_FS_HZ = 500.0


def _is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


@dataclass(frozen=True)
class ResampleContract:
    fs_source_hz: Optional[float]
    fs_internal_hz: float
    resample_applied: bool
    resample_method: str
    resample_ratio: Optional[str]
    fs_low_warning: bool
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _best_rational_ratio(fs_in: float, fs_out: float, max_den: int = 1000) -> Tuple[int, int]:
    """
    Return (up, down) such that fs_in * up / down ≈ fs_out.
    Uses a bounded rational approximation to keep FIR sizes reasonable.
    """
    # Use numpy rational approximation via fractions if available
    from fractions import Fraction

    frac = Fraction(fs_out / fs_in).limit_denominator(max_den)
    return int(frac.numerator), int(frac.denominator)


def resample_1d_to_target(
    x: np.ndarray,
    fs_in: float,
    *,
    fs_out: float = TARGET_FS_HZ,
    prefer_polyphase: bool = True,
) -> Tuple[np.ndarray, ResampleContract]:
    """
    Pure function:
      - Takes 1D signal x at fs_in
      - Returns signal at fs_out + contract dict

    Notes:
      - Upsampling does NOT create new clinical information; we track it explicitly.
      - If scipy is unavailable, we fall back to linear interpolation (marked in contract).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if x.size < 2:
        contract = ResampleContract(
            fs_source_hz=float(fs_in) if _is_finite(fs_in) else None,
            fs_internal_hz=float(fs_out),
            resample_applied=False,
            resample_method="none",
            resample_ratio=None,
            fs_low_warning=False,
            notes="signal_too_short",
        )
        return x.astype(np.float32, copy=False), contract

    if (not _is_finite(fs_in)) or (float(fs_in) <= 0):
        contract = ResampleContract(
            fs_source_hz=None,
            fs_internal_hz=float(fs_out),
            resample_applied=False,
            resample_method="none",
            resample_ratio=None,
            fs_low_warning=True,
            notes="bad_fs_in",
        )
        return x.astype(np.float32, copy=False), contract

    fs_in = float(fs_in)
    fs_out = float(fs_out)

    # If already at target within tolerance: no-op
    if abs(fs_in - fs_out) < 1e-6:
        contract = ResampleContract(
            fs_source_hz=fs_in,
            fs_internal_hz=fs_out,
            resample_applied=False,
            resample_method="none",
            resample_ratio=None,
            fs_low_warning=(fs_in < 250.0),
        )
        return x.astype(np.float32, copy=False), contract

    fs_low_warning = fs_in < 250.0  # normative-min heuristic; adjust if you want

    # Choose method
    if prefer_polyphase and _HAS_SCIPY and (resample_poly is not None):
        up, down = _best_rational_ratio(fs_in, fs_out, max_den=1000)
        # Reduce ratio if possible
        g = _gcd(up, down)
        up //= g
        down //= g

        y = resample_poly(x, up, down).astype(np.float32, copy=False)

        contract = ResampleContract(
            fs_source_hz=fs_in,
            fs_internal_hz=fs_out,
            resample_applied=True,
            resample_method="polyphase_fir",
            resample_ratio=f"{up}/{down}",
            fs_low_warning=fs_low_warning,
        )
        return y, contract

    # Fallback: linear interpolation (no scipy)
    n_out = int(round(x.size * (fs_out / fs_in)))
    n_out = max(n_out, 2)

    t_in = np.linspace(0.0, (x.size - 1) / fs_in, num=x.size, dtype=np.float64)
    t_out = np.linspace(0.0, (x.size - 1) / fs_in, num=n_out, dtype=np.float64)
    y = np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)

    contract = ResampleContract(
        fs_source_hz=fs_in,
        fs_internal_hz=fs_out,
        resample_applied=True,
        resample_method="linear_interp_fallback",
        resample_ratio=f"{n_out}/{x.size}",
        fs_low_warning=fs_low_warning,
        notes="scipy_unavailable",
    )
    return y, contract


def resample_leads_to_500hz(
    leads: Mapping[str, np.ndarray],
    fs_in: float,
    *,
    prefer_polyphase: bool = True,
) -> Tuple[Dict[str, np.ndarray], ResampleContract, Dict[str, Any]]:
    """
    Resample all leads to 500 Hz (internal standard).
    Returns:
      - leads_500: dict lead->np.ndarray (float32)
      - contract: ResampleContract (shared fs-level contract)
      - per_lead: dict lead-> {n_in, n_out} + optional notes
    """
    leads_500: Dict[str, np.ndarray] = {}
    per_lead: Dict[str, Any] = {}

    # Build a single contract at fs level (per-lead lengths will vary if inputs differ)
    # Use lead II if present; otherwise first non-empty lead to infer signal length behavior.
    sample_lead = None
    for k, v in leads.items():
        if v is not None and np.asarray(v).size >= 2:
            sample_lead = np.asarray(v, dtype=np.float32).reshape(-1)
            break

    if sample_lead is None:
        # No resampling possible; still emit contract
        contract = ResampleContract(
            fs_source_hz=float(fs_in) if _is_finite(fs_in) else None,
            fs_internal_hz=float(TARGET_FS_HZ),
            resample_applied=False,
            resample_method="none",
            resample_ratio=None,
            fs_low_warning=True,
            notes="no_valid_leads",
        )
        return {}, contract, {"per_lead": per_lead}

    # Resample sample lead to generate contract method/ratio
    _, contract = resample_1d_to_target(sample_lead, fs_in, fs_out=TARGET_FS_HZ, prefer_polyphase=prefer_polyphase)

    # Apply same method to each lead independently (lengths may differ)
    for name, arr in leads.items():
        if arr is None:
            continue
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size < 2:
            continue

        y, _c = resample_1d_to_target(x, fs_in, fs_out=TARGET_FS_HZ, prefer_polyphase=prefer_polyphase)
        leads_500[str(name)] = y
        per_lead[str(name)] = {"n_in": int(x.size), "n_out": int(y.size)}

    meta = {"per_lead": per_lead}
    return leads_500, contract, meta
    
# Terminus