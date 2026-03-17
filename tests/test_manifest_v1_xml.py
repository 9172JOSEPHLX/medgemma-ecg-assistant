# tests/test_manifest_v1_xml.py  MARS 16TH, 2026 V1.000

import xml.etree.ElementTree as ET
import numpy as np

from medgem_poc.input_adapters.ecg_digitization.convert import convert_ecg_image, ConvertConfig


def _fake_prob_map_provider(_path: str) -> np.ndarray:
    H, W = 64, 256
    pm = np.zeros((H, W), dtype=np.float32)
    pm[H // 2, :] = 1.0
    return pm


def test_manifest_extraction_qc_has_status_attr(tmp_path):
    out_dir = tmp_path / "out"
    cfg = ConvertConfig(fs_out=500, duration_s=1.0)
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

    man = out_dir / "manifest_v1.xml"
    assert man.exists()

    ns = {"mdg": "https://medgemma.ecg/manifest/v1"}
    root = ET.parse(str(man)).getroot()
    exq = root.find("mdg:extraction_qc", ns)
    assert exq is not None
    assert exq.attrib.get("status") == res["extraction_status"]

# TERMINUS