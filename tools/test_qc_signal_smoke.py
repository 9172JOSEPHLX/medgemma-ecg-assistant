# tools/test_qc_signal_smoke.py  Version QC.PY du 02.02.2026 ENRICHI DES BLOCS “QC clinique/mesures”

import medgem_poc.qc as qc

def assert_has_code(report, code: str):
    codes = [f["code"] for f in report.get("flags", [])]
    assert code in codes, f"Expected {code}, got {codes}"

def main() -> int:
    # FAIL low SNR
    m = {"snr": 3, "baseline_drift": 0.1, "filter_strength": 0.1}
    r = qc.qc_signal(m, patient_sex="M")
    assert r["status"] == "FAIL"
    assert_has_code(r, "FAIL_LOW_SNR")

    # WARN baseline drift
    m = {"snr": 12, "baseline_drift": 0.7, "filter_strength": 0.1}
    r = qc.qc_signal(m, patient_sex="M")
    assert r["status"] == "WARN"
    assert_has_code(r, "WARN_BASELINE_DRIFT")

    # WARN filter suspected
    m = {"snr": 12, "baseline_drift": 0.1, "filter_strength": 0.9}
    r = qc.qc_signal(m, patient_sex="M")
    assert r["status"] == "WARN"
    assert_has_code(r, "WARN_FILTER_APPLIED")

    # WARN QRS wide
    m = {"snr": 12, "baseline_drift": 0.1, "filter_strength": 0.1, "qrs_ms": 140}
    r = qc.qc_signal(m, patient_sex="M")
    assert r["status"] == "WARN"
    assert_has_code(r, "WARN_QRS_BROAD")

    print("qc_signal smoke tests: OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

# Terminus