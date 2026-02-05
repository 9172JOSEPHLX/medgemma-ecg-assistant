import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medgem_poc.qc import load_image_bgr, qc_gate, qc_to_json, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to ECG image (png/jpg)")
    ap.add_argument("--out", default="", help="Optional output qc JSON path")
    args = ap.parse_args()

    img = load_image_bgr(args.image)
    qc = qc_gate(img)

    print(qc_to_json(qc))

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        save_json(str(outp), qc.__dict__)


if __name__ == "__main__":
    main()
