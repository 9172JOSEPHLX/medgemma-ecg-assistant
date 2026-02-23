# tools\make_ra_la_swap.py FEB 23 th, 2026

#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

EXPECTED_COLS = ["time_s","I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

def ra_la_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    RA <-> LA limb electrode reversal (Taboulet / standard mapping):
      I -> -I
      II <-> III
      aVR <-> aVL
      aVF unchanged
      V1-V6 unchanged
    Input/Output schema preserved: 1250x13 (time_s + 12 leads)
    """
    d = df.copy()

    # keep originals
    I_old   = d["I"].copy()
    II_old  = d["II"].copy()
    III_old = d["III"].copy()
    aVR_old = d["aVR"].copy()
    aVL_old = d["aVL"].copy()

    d["I"]   = -I_old
    d["II"]  = III_old
    d["III"] = II_old
    d["aVR"] = aVL_old
    d["aVL"] = aVR_old
    # aVF unchanged; V1-V6 unchanged

    return d

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate RA<->LA swapped ECG CSV (time_s + 12 leads).")
    ap.add_argument("--src", required=True, help="Input ECG CSV path (time_s + 12 leads).")
    ap.add_argument("--dst", required=True, help="Output CSV path.")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise SystemExit(f"Source file not found: {src}")

    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}\nFound columns: {list(df.columns)}")

    df = df[EXPECTED_COLS]  # enforce exact order
    out = ra_la_swap(df)
    out = out[EXPECTED_COLS]

    out.to_csv(dst, index=False)

    print("✅ wrote:", dst)
    print("shape:", out.shape)
    print("cols :", list(out.columns))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    
# Terminus