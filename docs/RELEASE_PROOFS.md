# Release Proofs

This repository uses the immutable tag below as the jury/source-of-truth code reference:

- Tag: `v0.1.1-kaggle-jury-20260312`
- Commit: `49de9a0`

This tag must remain immutable.

## Purpose

The proof bundle is a deterministic release artifact intended for:
- jury/audit review,
- offline inspection,
- traceability of the validated ECG QC baseline.

It packages:
- tagged source files from the immutable tag,
- jury-oriented documentation,
- locally generated proof artifacts when available.

## Source-of-truth rule

For any jury-facing verification:
1. use the immutable tag,
2. use the tagged source files exported from that tag,
3. use the proof bundle manifest and hashes.

Do not use `main` HEAD as the primary proof reference.

## Build the proof bundle

From repository root:

```bash
python scripts/build_proofs_bundle.py --repo-root . --tag v0.1.1-kaggle-jury-20260312