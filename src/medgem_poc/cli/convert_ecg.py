# file: src/medgem_poc/cli/convert_ecg.py  PRJ APPLI MEDGEMMA ECG 23/03/2026

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from medgem_poc.input_adapters.ecg_digitization import (
    ConvertConfig,
    RegressionInferConfig,
    convert_ecg_image,
)

EXIT_OK = 0
EXIT_INVALID_CONFIG = 3
EXIT_INPUT_NOT_FOUND = 4
EXIT_PROB_MAP_NOT_FOUND = 5
EXIT_CKPT_NOT_FOUND = 6
EXIT_RUNTIME_ERROR = 7


def _err(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def _parse_input_hw(value: str) -> tuple[int, int]:
    text = str(value).strip().lower().replace(" ", "")
    if "x" not in text:
        raise ValueError("--input_hw must be like 256x512")
    h_str, w_str = text.split("x", 1)
    try:
        h = int(h_str)
        w = int(w_str)
    except ValueError as exc:
        raise ValueError("--input_hw must be like 256x512") from exc
    if h <= 0 or w <= 0:
        raise ValueError("--input_hw must be positive, e.g. 256x512")
    return h, w


def _parse_coord_channels(value: int) -> bool:
    if value not in (0, 1):
        raise ValueError("--coord_channels must be 0 or 1")
    return bool(value)


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_non_negative_int(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_float(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_optional_positive_float(name: str, value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be > 0 when provided")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medgem_poc.cli.convert_ecg",
        description=(
            "Convert an ECG image into 12-lead CSV + manifest XML using either "
            "a precomputed prob_map.npy or a TorchScript regression checkpoint."
        ),
    )
    parser.add_argument("--in", dest="inp", required=True, help="Input ECG image path")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    parser.add_argument("--page", type=int, default=0, help="Page index for multipage inputs")
    parser.add_argument("--dpi", type=int, default=400, help="Render DPI")
    parser.add_argument("--fs", type=int, default=500, help="Output sampling frequency in Hz")
    parser.add_argument("--duration_s", type=float, default=10.0, help="Output duration in seconds")
    parser.add_argument("--px_per_mm", type=float, default=None, help="Calibration: pixels per mm")
    parser.add_argument("--ckpt", type=str, default="", help="TorchScript checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="cpu | cuda | None")
    parser.add_argument("--input_hw", type=str, default="256x512", help="HxW, e.g. 1280x2528")
    parser.add_argument("--coord_channels", type=int, default=0, help="0 or 1")
    parser.add_argument("--prob_map_npy", type=str, default="", help="Precomputed prob_map .npy path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        _validate_non_negative_int("--page", args.page)
        _validate_positive_int("--dpi", args.dpi)
        _validate_positive_int("--fs", args.fs)
        _validate_positive_float("--duration_s", float(args.duration_s))
        _validate_optional_positive_float("--px_per_mm", args.px_per_mm)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_INVALID_CONFIG

    input_path = Path(args.inp)
    if not input_path.is_file():
        _err(f"input file not found: {input_path}")
        return EXIT_INPUT_NOT_FOUND

    prob_map_npy = str(args.prob_map_npy or "").strip()
    ckpt = str(args.ckpt or "").strip()

    if not prob_map_npy and not ckpt:
        _err("provide either --prob_map_npy <file.npy> or --ckpt <model.ts>")
        return EXIT_INVALID_CONFIG

    if prob_map_npy and not Path(prob_map_npy).is_file():
        _err(f"prob_map_npy file not found: {prob_map_npy}")
        return EXIT_PROB_MAP_NOT_FOUND

    reg_cfg = None
    if ckpt:
        if not Path(ckpt).is_file():
            _err(f"ckpt file not found: {ckpt}")
            return EXIT_CKPT_NOT_FOUND

        try:
            input_hw = _parse_input_hw(args.input_hw)
            coord_channels = _parse_coord_channels(int(args.coord_channels))
        except ValueError as exc:
            _err(str(exc))
            return EXIT_INVALID_CONFIG

        reg_cfg = RegressionInferConfig(
            ckpt_path=ckpt,
            device=args.device,
            input_hw=input_hw,
            add_coord_channels=coord_channels,
        )

    cfg = ConvertConfig(
        fs_out=args.fs,
        duration_s=float(args.duration_s),
    )

    try:
        result = convert_ecg_image(
            input_path=str(input_path),
            out_dir=args.out_dir,
            cfg=cfg,
            page=args.page,
            dpi=args.dpi,
            px_per_mm_mean=args.px_per_mm,
            regression_cfg=reg_cfg,
            prob_map_npy=(prob_map_npy if prob_map_npy else None),
        )
    except Exception as exc:
        _err(f"convert_ecg failed: {exc}")
        return EXIT_RUNTIME_ERROR

    print(f"CSV: {result['paths']['csv']} sha256={result['sha256']['csv']}")
    print(f"XML: {result['paths']['manifest']} sha256={result['sha256']['manifest']}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus
