# file: tests/test_convert_ecg_cli.py      PRJ APPLI MEDGEMMA ECG 23/03/2026

from __future__ import annotations

from pathlib import Path

import numpy as np

import medgem_poc.cli.convert_ecg as cli


def _write_dummy_input(tmp_path: Path) -> Path:
    p = tmp_path / "dummy.png"
    p.write_bytes(b"not-a-real-png")
    return p


def _write_dummy_prob_map(tmp_path: Path) -> Path:
    p = tmp_path / "pm.npy"
    np.save(p, np.zeros((8, 8), dtype=np.float32))
    return p


def test_cli_missing_input_file_returns_4(capsys, tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    pm = _write_dummy_prob_map(tmp_path)

    rc = cli.main(
        [
            "--in",
            str(tmp_path / "missing.png"),
            "--out",
            str(out_dir),
            "--prob_map_npy",
            str(pm),
        ]
    )

    err = capsys.readouterr().err
    assert rc == cli.EXIT_INPUT_NOT_FOUND
    assert "input file not found" in err


def test_cli_requires_prob_map_or_ckpt(capsys, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"

    rc = cli.main(["--in", str(inp), "--out", str(out_dir)])

    err = capsys.readouterr().err
    assert rc == cli.EXIT_INVALID_CONFIG
    assert "provide either --prob_map_npy" in err


def test_cli_missing_prob_map_file_returns_5(capsys, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--in",
            str(inp),
            "--out",
            str(out_dir),
            "--prob_map_npy",
            str(tmp_path / "missing.npy"),
        ]
    )

    err = capsys.readouterr().err
    assert rc == cli.EXIT_PROB_MAP_NOT_FOUND
    assert "prob_map_npy file not found" in err


def test_cli_missing_ckpt_file_returns_6(capsys, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"

    rc = cli.main(
        [
            "--in",
            str(inp),
            "--out",
            str(out_dir),
            "--ckpt",
            str(tmp_path / "missing.ts"),
        ]
    )

    err = capsys.readouterr().err
    assert rc == cli.EXIT_CKPT_NOT_FOUND
    assert "ckpt file not found" in err


def test_cli_invalid_input_hw_returns_3(capsys, monkeypatch, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"
    ckpt = tmp_path / "model.ts"
    ckpt.write_bytes(b"fake")

    rc = cli.main(
        [
            "--in",
            str(inp),
            "--out",
            str(out_dir),
            "--ckpt",
            str(ckpt),
            "--input_hw",
            "bad-shape",
        ]
    )

    err = capsys.readouterr().err
    assert rc == cli.EXIT_INVALID_CONFIG
    assert "--input_hw must be like 256x512" in err


def test_cli_runtime_error_returns_7(capsys, monkeypatch, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"
    pm = _write_dummy_prob_map(tmp_path)

    def _boom(**_: object) -> dict[str, object]:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "convert_ecg_image", _boom)

    rc = cli.main(
        [
            "--in",
            str(inp),
            "--out",
            str(out_dir),
            "--prob_map_npy",
            str(pm),
        ]
    )

    err = capsys.readouterr().err
    assert rc == cli.EXIT_RUNTIME_ERROR
    assert "convert_ecg failed: boom" in err


def test_cli_success_prints_two_proof_lines(capsys, monkeypatch, tmp_path: Path) -> None:
    inp = _write_dummy_input(tmp_path)
    out_dir = tmp_path / "out"
    pm = _write_dummy_prob_map(tmp_path)

    def _fake_convert(**_: object) -> dict[str, object]:
        return {
            "paths": {
                "csv": str(out_dir / "ecg_12lead_500hz.csv"),
                "manifest": str(out_dir / "manifest_v1.xml"),
            },
            "sha256": {
                "csv": "abc123",
                "manifest": "def456",
            },
        }

    monkeypatch.setattr(cli, "convert_ecg_image", _fake_convert)

    rc = cli.main(
        [
            "--in",
            str(inp),
            "--out",
            str(out_dir),
            "--prob_map_npy",
            str(pm),
        ]
    )

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]

    assert rc == cli.EXIT_OK
    assert len(lines) == 2
    assert lines[0].startswith("CSV: ")
    assert "sha256=abc123" in lines[0]
    assert lines[1].startswith("XML: ")
    assert "sha256=def456" in lines[1]

# Terminus