# file: scripts/build_proofs_bundle.py     23/03/2026    PRJ APPLI MEDGEMMA ECG  V1.00 12H40
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_TAG = "v0.1.1-kaggle-jury-20260312"
DEFAULT_PROOF_FILES = [
    "docs/JURY_APPENDIX.md",
    "Logs/proofs_2026-03-12/bench_swap_results.json",
    "Logs/proofs_2026-03-12/bench_swap_summary.csv",
]

TAGGED_SOURCE_FILES = [
    "src/medgem_poc/qc.py",
    "src/medgem_poc/edge_metrics.py",
]


@dataclass(frozen=True)
class BundleItem:
    src_path: Path
    arcname: str
    sha256: str
    kind: str


def _run_git(args: list[str], repo_root: Path) -> bytes:
    p = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=False,
    )
    return p.stdout


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_tagged_file(repo_root: Path, tag: str, rel_path: str, out_path: Path) -> BundleItem:
    content = _run_git(["show", f"{tag}:{rel_path}"], repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content)
    return BundleItem(
        src_path=out_path,
        arcname=str(out_path.relative_to(out_path.parents[1])).replace("\\", "/"),
        sha256=_sha256_bytes(content),
        kind="tagged_source",
    )


def _copy_existing_file(repo_root: Path, rel_path: str, out_root: Path) -> BundleItem | None:
    src = repo_root / rel_path
    if not src.exists():
        return None
    dst = out_root / "working_tree" / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return BundleItem(
        src_path=dst,
        arcname=str(dst.relative_to(out_root)).replace("\\", "/"),
        sha256=_sha256_file(dst),
        kind="working_tree_file",
    )


def _iter_existing(items: Iterable[BundleItem | None]) -> list[BundleItem]:
    return [x for x in items if x is not None]


def build_bundle(repo_root: Path, tag: str, out_dir: Path, extra_files: list[str]) -> Path:
    tag_out = out_dir / tag
    tag_out.mkdir(parents=True, exist_ok=True)

    manifest_items: list[BundleItem] = []

    tagged_root = tag_out / "tagged_source"
    for rel_path in TAGGED_SOURCE_FILES:
        dst = tagged_root / rel_path
        manifest_items.append(_export_tagged_file(repo_root, tag, rel_path, dst))

    working_items = _iter_existing(
        _copy_existing_file(repo_root, rel_path, tag_out)
        for rel_path in [*DEFAULT_PROOF_FILES, *extra_files]
    )
    manifest_items.extend(working_items)

    head = _run_git(["rev-parse", "HEAD"], repo_root).decode("utf-8", errors="replace").strip()
    branch = _run_git(["branch", "--show-current"], repo_root).decode("utf-8", errors="replace").strip()

    manifest = {
        "tag": tag,
        "repo_root": str(repo_root),
        "branch": branch,
        "head": head,
        "items": [
            {
                "kind": item.kind,
                "arcname": item.arcname,
                "sha256": item.sha256,
            }
            for item in sorted(manifest_items, key=lambda x: x.arcname)
        ],
        "required_tagged_files": TAGGED_SOURCE_FILES,
        "default_optional_files": DEFAULT_PROOF_FILES,
    }

    manifest_path = tag_out / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest_items.append(
        BundleItem(
            src_path=manifest_path,
            arcname="MANIFEST.json",
            sha256=_sha256_file(manifest_path),
            kind="bundle_manifest",
        )
    )

    zip_path = out_dir / f"{tag}-proofs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(manifest_items, key=lambda x: x.arcname):
            zf.write(item.src_path, arcname=item.arcname)

    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic jury proof bundle ZIP.")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Immutable source-of-truth tag")
    parser.add_argument("--out-dir", default="release_assets", help="Output directory")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra repo-relative file to include in the bundle",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not (repo_root / ".git").exists():
        print(f"ERROR: not a git repo: {repo_root}", file=sys.stderr)
        return 2

    try:
        _run_git(["rev-parse", "--verify", args.tag], repo_root)
    except subprocess.CalledProcessError:
        print(f"ERROR: tag not found: {args.tag}", file=sys.stderr)
        return 3

    zip_path = build_bundle(repo_root, args.tag, out_dir, args.include)
    print(f"BUNDLE_ZIP={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Terminus