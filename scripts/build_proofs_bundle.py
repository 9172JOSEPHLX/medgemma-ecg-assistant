# file: scripts/build_proofs_bundle.py  
from __future__ import annotations

import argparse
import hashlib
import json
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

# Fixed timestamp for more stable ZIP metadata.
# ZIP format does not support dates before 1980.
ZIP_FIXED_DT = (2026, 3, 12, 0, 0, 0)


@dataclass(frozen=True)
class BundleItem:
    src_path: Path
    arcname: str
    sha256: str
    kind: str


@dataclass(frozen=True)
class BundleResult:
    zip_path: Path
    zip_sha256: str
    manifest_path: Path
    bundle_info_path: Path
    missing_files: list[str]
    tag: str
    tag_commit: str
    branch: str
    head: str


class BundleBuildError(RuntimeError):
    pass


def _run_git_bytes(args: list[str], repo_root: Path) -> bytes:
    p = subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        check=True,
        capture_output=True,
        text=False,
    )
    return p.stdout


def _run_git_text(args: list[str], repo_root: Path) -> str:
    return _run_git_bytes(args, repo_root).decode("utf-8", errors="replace").strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_rel_path(rel_path: str) -> str:
    return str(Path(rel_path)).replace("\\", "/")


def _unique_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        norm = _normalize_rel_path(p)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _require_git_repo(repo_root: Path) -> None:
    if not (repo_root / ".git").exists():
        raise BundleBuildError(f"not a git repo: {repo_root}")


def _require_tag_exists(repo_root: Path, tag: str) -> None:
    try:
        _run_git_text(["rev-parse", "--verify", tag], repo_root)
    except subprocess.CalledProcessError as e:
        raise BundleBuildError(f"tag not found: {tag}") from e


def _resolve_commit(repo_root: Path, ref: str) -> str:
    try:
        return _run_git_text(["rev-list", "-n", "1", ref], repo_root)
    except subprocess.CalledProcessError as e:
        raise BundleBuildError(f"cannot resolve commit for ref: {ref}") from e


def _export_tagged_file(repo_root: Path, tag: str, rel_path: str, out_root: Path) -> BundleItem:
    rel_path = _normalize_rel_path(rel_path)
    try:
        content = _run_git_bytes(["show", f"{tag}:{rel_path}"], repo_root)
    except subprocess.CalledProcessError as e:
        raise BundleBuildError(f"required tagged file not found in tag '{tag}': {rel_path}") from e

    dst = out_root / "tagged_source" / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(content)

    return BundleItem(
        src_path=dst,
        arcname=str(dst.relative_to(out_root)).replace("\\", "/"),
        sha256=_sha256_bytes(content),
        kind="tagged_source",
    )


def _copy_working_tree_file(
    repo_root: Path,
    rel_path: str,
    out_root: Path,
) -> BundleItem | None:
    rel_path = _normalize_rel_path(rel_path)
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


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _zipinfo_for_arcname(arcname: str) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(filename=arcname, date_time=ZIP_FIXED_DT)
    zi.compress_type = zipfile.ZIP_DEFLATED
    # 0o100644 regular file permissions on Unix-like systems
    zi.external_attr = 0o100644 << 16
    return zi


def _write_zip_deterministic(items: list[BundleItem], zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for item in sorted(items, key=lambda x: x.arcname):
            data = item.src_path.read_bytes()
            zf.writestr(_zipinfo_for_arcname(item.arcname), data)


def build_bundle(
    repo_root: Path,
    tag: str,
    out_dir: Path,
    extra_files: list[str],
    *,
    strict: bool = False,
    expected_tag_commit: str | None = None,
) -> BundleResult:
    _require_git_repo(repo_root)
    _require_tag_exists(repo_root, tag)

    tag_commit = _resolve_commit(repo_root, tag)
    if expected_tag_commit is not None and tag_commit != expected_tag_commit:
        raise BundleBuildError(
            f"tag '{tag}' points to {tag_commit}, expected {expected_tag_commit}"
        )

    tag_out = out_dir / tag
    tag_out.mkdir(parents=True, exist_ok=True)

    manifest_items: list[BundleItem] = []

    # 1) Required tagged source-of-truth files
    for rel_path in TAGGED_SOURCE_FILES:
        manifest_items.append(_export_tagged_file(repo_root, tag, rel_path, tag_out))

    # 2) Optional/extra working-tree files
    requested_working_files = _unique_paths([*DEFAULT_PROOF_FILES, *extra_files])
    missing_files: list[str] = []

    working_items: list[BundleItem] = []
    for rel_path in requested_working_files:
        item = _copy_working_tree_file(repo_root, rel_path, tag_out)
        if item is None:
            missing_files.append(rel_path)
        else:
            working_items.append(item)
    manifest_items.extend(working_items)

    if strict and missing_files:
        missing_str = "\n".join(f"  - {p}" for p in missing_files)
        raise BundleBuildError(
            "strict mode failed: missing requested proof file(s):\n" + missing_str
        )

    head = _resolve_commit(repo_root, "HEAD")
    branch = _run_git_text(["branch", "--show-current"], repo_root)

    manifest = {
        "tag": tag,
        "tag_commit": tag_commit,
        "expected_tag_commit": expected_tag_commit,
        "repo_root": str(repo_root),
        "branch": branch,
        "head": head,
        "strict_mode": strict,
        "required_tagged_files": TAGGED_SOURCE_FILES,
        "requested_working_tree_files": requested_working_files,
        "missing_working_tree_files": missing_files,
        "items": [
            {
                "kind": item.kind,
                "arcname": item.arcname,
                "sha256": item.sha256,
            }
            for item in sorted(manifest_items, key=lambda x: x.arcname)
        ],
    }

    manifest_path = tag_out / "MANIFEST.json"
    _write_json(manifest_path, manifest)

    manifest_item = BundleItem(
        src_path=manifest_path,
        arcname="MANIFEST.json",
        sha256=_sha256_file(manifest_path),
        kind="bundle_manifest",
    )
    manifest_items.append(manifest_item)

    zip_path = out_dir / f"{tag}-proofs.zip"
    _write_zip_deterministic(manifest_items, zip_path)
    zip_sha256 = _sha256_file(zip_path)

    bundle_info = {
        "tag": tag,
        "tag_commit": tag_commit,
        "expected_tag_commit": expected_tag_commit,
        "branch": branch,
        "head": head,
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha256,
        "manifest_path": str(manifest_path),
        "strict_mode": strict,
        "missing_working_tree_files": missing_files,
    }
    bundle_info_path = out_dir / f"{tag}-BUNDLE_INFO.json"
    _write_json(bundle_info_path, bundle_info)

    return BundleResult(
        zip_path=zip_path,
        zip_sha256=zip_sha256,
        manifest_path=manifest_path,
        bundle_info_path=bundle_info_path,
        missing_files=missing_files,
        tag=tag,
        tag_commit=tag_commit,
        branch=branch,
        head=head,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build jury proof bundle ZIP for medgemma-ecg-assistant."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Immutable source-of-truth tag")
    parser.add_argument("--out-dir", default="release_assets", help="Output directory")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Extra repo-relative file to include in the bundle (repeatable)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested working-tree proof file is missing",
    )
    parser.add_argument(
        "--expected-tag-commit",
        default=None,
        help="Fail unless --tag resolves exactly to this commit SHA",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    try:
        result = build_bundle(
            repo_root=repo_root,
            tag=args.tag,
            out_dir=out_dir,
            extra_files=args.include,
            strict=args.strict,
            expected_tag_commit=args.expected_tag_commit,
        )
    except BundleBuildError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as e:
        print(f"ERROR: git command failed: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(f"ERROR: unexpected failure: {e}", file=sys.stderr)
        return 4

    if result.missing_files:
        print("WARNING: some requested proof files were missing:", file=sys.stderr)
        for rel_path in result.missing_files:
            print(f"  - {rel_path}", file=sys.stderr)

    print(f"BUNDLE_TAG={result.tag}")
    print(f"BUNDLE_TAG_COMMIT={result.tag_commit}")
    print(f"BUNDLE_BRANCH={result.branch}")
    print(f"BUNDLE_HEAD={result.head}")
    print(f"BUNDLE_MANIFEST={result.manifest_path}")
    print(f"BUNDLE_INFO={result.bundle_info_path}")
    print(f"BUNDLE_ZIP={result.zip_path}")
    print(f"BUNDLE_ZIP_SHA256={result.zip_sha256}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
