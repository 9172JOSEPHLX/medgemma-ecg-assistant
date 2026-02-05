# tools/test_qc_calibration.py   Version QC.PY du 02.02.2026 ENRICHI DES BLOCS “QC clinique/mesures”

import medgem_poc.qc as qc

def main() -> int:
    assert qc.SPEED_MM_S == 25, f"SPEED_MM_S must be 25, got {qc.SPEED_MM_S}"
    assert qc.GAIN_MM_MV == 10, f"GAIN_MM_MV must be 10, got {qc.GAIN_MM_MV}"

    assert qc.SMALL_SQUARE_MS == 40, f"SMALL_SQUARE_MS must be 40, got {qc.SMALL_SQUARE_MS}"
    assert qc.LARGE_SQUARE_MS == 200, f"LARGE_SQUARE_MS must be 200, got {qc.LARGE_SQUARE_MS}"

    # Derived sanity
    assert 5 * qc.SMALL_SQUARE_MS == qc.LARGE_SQUARE_MS, "5 small squares must equal 1 large square"
    assert 5 * qc.LARGE_SQUARE_MS == 1000, "5 large squares must equal 1 second at 25 mm/s"

    print("Calibration constants: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# Terminus