import numpy as np
from medgem_poc.input_adapters.ecg_digitization.convert import convert_ecg_image, ConvertConfig

def _fake_prob_map_provider(_path: str) -> np.ndarray:
    H, W = 128, 512
    pm = np.zeros((H, W), dtype=np.float32)
    for x in range(W):
        y = int((H * 0.35) + (H * 0.15) * np.sin(2*np.pi*x/80.0))
        y = max(0, min(H-1, y))
        pm[y, x] = 1.0
        if y+1 < H: pm[y+1, x] = 0.6
        if y-1 >= 0: pm[y-1, x] = 0.6
    return np.clip(pm, 0.0, 1.0)

def test_convert_smoke(tmp_path):
    out_dir = tmp_path / "out"
    cfg = ConvertConfig(fs_out=500, duration_s=2.0)  # quick
    res = convert_ecg_image(
        input_path=str(tmp_path / "dummy.png"),
        out_dir=str(out_dir),
        cfg=cfg,
        px_per_mm_mean=10.0,
        rectification_confidence=1.0,
        calibration_confidence=1.0,
        layout_confidence=1.0,
        prob_map_provider=_fake_prob_map_provider,
    )
    assert res["extraction_status"] in ("EXTRACT_PASS", "EXTRACT_WARN", "EXTRACT_FAIL")
    assert (out_dir / "ecg_12lead_500hz.csv").exists()
    assert (out_dir / "manifest_v1.xml").exists()
    m = res["metrics"]
    assert m["lead_count"] == 12
    assert np.isfinite(m["trace_coverage_pct"])
    assert np.isfinite(m["gap_max_ms"])
