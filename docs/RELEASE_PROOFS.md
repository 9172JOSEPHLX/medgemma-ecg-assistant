# Release Proofs

## Purpose

This document defines the standard process for generating, verifying, and publishing jury-safe release proof bundles for **Appli_MedGemma_ECG**.

It complements:
- `docs/BRANCH_POLICY.md` for repository governance,
- `.github/pull_request_template.md` for PR disclosure and validation,
- `scripts/build_proofs_bundle.py` for bundle generation,
- `.gitignore` for local artifact hygiene.

This document does **not** redefine branch policy or runtime invariants in full. It explains how proof artifacts must be built and handled during release closure.

---

## Immutable jury reference

The immutable jury reference is:

- **Tag:** `v0.1.1-kaggle-jury-20260312`
- **Commit:** `49de9a0`

Any jury-safe proof bundle must be generated against this immutable source-of-truth unless an explicit future release process states otherwise.

---

## Standard tool

The standard tool for proof bundle generation is:

- `scripts/build_proofs_bundle.py`

The builder is responsible for:
- exporting the tagged runtime source-of-truth files,
- collecting working-tree proof artifacts,
- writing `MANIFEST.json`,
- generating the release ZIP,
- producing bundle metadata including `BUNDLE_ZIP_SHA256`.

---

## Standard command

Use the following command as the default safe release-proof command:

```bash
python scripts/build_proofs_bundle.py \
  --tag v0.1.1-kaggle-jury-20260312 \
  --expected-tag-commit 49de9a0 \
  --strict
