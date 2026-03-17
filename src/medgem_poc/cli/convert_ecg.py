from __future__ import annotations

import argparse
import os

from medgem_poc.input_adapters.ecg_digitization import ConvertConfig, RegressionInferConfig, convert_ecg_image


def main() -> None:
    p = argparse.ArgumentParser("medgem_poc.cli.convert_ecg")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out_dir", required=True)
    p.add_argument("--page", type=int, default=0)
    p.add_argument("--dpi", type=int, default=400)

    p.add_argument("--fs", type=int, default=500)
    p.add_argument("--duration_s", type=float, default=10.0)
    p.add_argument("--px_per_mm", type=float, default=None)

    # Regression (torchscript)
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--device", type=str, default=None, help="cpu|cuda|None")
    p.add_argument("--input_hw", type=str, default="256x512", help="HxW, e.g. 1280x2528")
    p.add_argument("--coord_channels", type=int, default=0, help="1: add X/Y coord channels")

    # Debug: use a precomputed prob_map.npy
    p.add_argument("--prob_map_npy", type=str, default="")

    a = p.parse_args()

    cfg = ConvertConfig(fs_out=a.fs, duration_s=a.duration_s)
    reg_cfg = None
    if a.ckpt:
        hs = str(a.input_hw).lower().replace(" ", "")
        if "x" not in hs:
            raise SystemExit("--input_hw must be like 256x512")
        h, w = hs.split("x", 1)
        reg_cfg = RegressionInferConfig(
            ckpt_path=a.ckpt,
            device=a.device,
            input_hw=(int(h), int(w)),
            add_coord_channels=bool(int(a.coord_channels)),
        )

    res = convert_ecg_image(
        input_path=a.inp,
        out_dir=a.out_dir,
        cfg=cfg,
        page=a.page,
        dpi=a.dpi,
        px_per_mm_mean=a.px_per_mm,
        regression_cfg=reg_cfg,
        prob_map_npy=(a.prob_map_npy if a.prob_map_npy else None),
    )

    # EXACT 2 proof lines
    print(f"CSV: {res['paths']['csv']} sha256={res['sha256']['csv']}")
    print(f"XML: {res['paths']['manifest']} sha256={res['sha256']['manifest']}")


if __name__ == "__main__":
    main()
