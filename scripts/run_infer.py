import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medgem_poc.qc import load_image_bgr, qc_gate
from medgem_poc.report_schema import ECGReport, ReportMeta, QCInfo


def medgemma_placeholder_summary(qc_level: str) -> str:
    if qc_level == "WARN":
        return "Interprétation assistée possible, mais la qualité de capture est moyenne. Validation clinicien renforcée."
    return "Interprétation assistée générée localement. Validation par clinicien requise."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to ECG image (png/jpg)")
    ap.add_argument("--out", default="output_report.json", help="Output report JSON path")
    ap.add_argument("--device", default="unknown", help="Device label (e.g., Galaxy A54)")
    ap.add_argument("--mode_avion", default="1", help="1 if airplane mode used for demo")
    args = ap.parse_args()

    img = load_image_bgr(args.image)
    qc = qc_gate(img)

    meta = ReportMeta(
        model_id="medgemma_placeholder",
        model_version="0.0",
        device=args.device,
        mode_avion=(str(args.mode_avion).strip() != "0"),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )

    qc_info = QCInfo(
        qc_level=qc.qc_level,
        qc_score=qc.qc_score,
        reasons=qc.reasons,
        metrics=qc.metrics,
        retake_prompt=qc.retake_prompt,
    )

    if qc.qc_level == "FAIL":
        report = ECGReport(
            meta=meta,
            qc=qc_info,
            summary="QC insuffisant : retake recommandé. Basculer sur lecture manuelle / protocole clinique.",
            confidence="LOW",
            clinician_checklist=[
                "Reprendre une capture (sans reflets, nette, cadrage complet).",
                "Si urgence et retake impossible : interprétation clinique standard sans assistance IA.",
            ],
            debug={"note": "Inference blocked due to QC_FAIL"},
        )
    else:
        report = ECGReport(
            meta=meta,
            qc=qc_info,
            summary=medgemma_placeholder_summary(qc.qc_level),
            confidence="LOW" if qc.qc_level == "WARN" else "MEDIUM",
            clinician_checklist=[
                "Vérifier rythme, FC, PR, QRS, QT sur l’ECG original.",
                "Confirmer toute suspicion (ST/T, blocs, arythmies) selon protocoles.",
            ],
            debug={"note": "MedGemma not wired yet (placeholder output)"},
        )

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(report.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report -> {outp}")


if __name__ == "__main__":
    main()
