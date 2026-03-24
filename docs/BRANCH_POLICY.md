# Branch Policy

## Purpose

This document defines the repository branch safety rules for **Appli_MedGemma_ECG**.

The goal is to:
- preserve the immutable jury-validated baseline,
- keep `main` stable, reviewable, and releasable,
- allow safe maintenance work to flow into `main`,
- keep post-jury feature development isolated until explicitly promoted.

This policy is operationally enforced through:
- `pull_request_template.md` for PR disclosure and validation,
- `scripts/build_proofs_bundle.py` for proof bundle generation,
- `.gitignore` for local artifact hygiene.

---

## Immutable source-of-truth

The immutable jury reference is:

- **Tag:** `v0.1.1-kaggle-jury-20260312`
- **Commit:** `49de9a0`

This tag must never be:
- modified,
- moved,
- replaced,
- force-updated.

Any release/proof workflow must treat this tag as the final jury code truth.

---

## Branch roles

### `main`
**Purpose**
- jury-safe branch,
- stable branch,
- reviewable branch,
- releasable branch.

**Allowed changes**
- documentation,
- release notes,
- jury appendix,
- PR template,
- branch policy,
- `.gitignore`,
- maintenance scripts that do **not** alter validated runtime behavior,
- release/proof tooling that does **not** alter validated runtime behavior.

**Forbidden during jury-safe closure**
- direct pushes,
- unreviewed runtime changes,
- post-jury feature merges,
- `dev/post-jury -> main`,
- changes that alter validated QC/runtime semantics without explicit re-baselining.

### `dev/maintenance-jury-safe`
**Purpose**
- safe staging branch for changes acceptable for `main`.

**Typical examples**
- docs updates,
- PR templates,
- release proof scripts,
- repo hygiene,
- non-runtime maintenance.

This is the **preferred source branch** for safe PRs into `main`.

### `dev/post-jury`
**Purpose**
- post-jury feature development,
- digitization MVP/productization,
- runtime evolution,
- broader technical consolidation and R&D.

**Typical examples**
- digitization adapters,
- OpenCV digitization work,
- CLI feature work,
- OCR/TorchScript integration,
- MedGemma workflow evolution beyond the jury-safe baseline.

This branch must **not** be merged into `main` during jury-safe closure.

---

## Pull request rules

### Allowed PRs to `main`
Allowed:
- `dev/maintenance-jury-safe -> main`
- dedicated docs-only branches -> `main`
- dedicated safe maintenance branches -> `main`

Not allowed during jury-safe closure:
- `dev/post-jury -> main`

### PR requirements
Every PR targeting `main` must:
- use the repository `pull_request_template.md`,
- have atomic scope,
- avoid unrelated file changes,
- clearly state whether runtime code under `src/` is touched,
- explicitly state whether jury-safe runtime invariants are preserved,
- explicitly reference the immutable tag when relevant,
- include validation commands and observed results.

If a PR touches runtime behavior, it is **not** considered maintenance-only.

---

## Runtime protection rules

The following are non-negotiable unless an explicit re-baselining decision is made:

- do **not** break the QC stable contract,
- do **not** change baseline semantics:
  - `09487_ok` remains `PASS / PASS / PASS`
  - `11899` remains `WARN / WARN / WARN`
  - limb swap remains acquisition warning only and does **not** become `FAIL` by itself
- do **not** modify the immutable tag,
- do **not** silently widen the role of MedGemma beyond the approved QC-gated diagnostic-aid posture.

If runtime code under `src/` is touched, the PR must explicitly explain:
- what changed,
- why it is safe,
- how the invariants were checked,
- what tests were run.

---

## Proof and release policy

### Standard proof bundle tool
The standard tool for jury proof bundle generation is:

- `scripts/build_proofs_bundle.py`

### Standard safe bundle command
When generating a jury-safe proof bundle, use the standard strict form:

```bash
python scripts/build_proofs_bundle.py \
  --tag v0.1.1-kaggle-jury-20260312 \
  --expected-tag-commit 49de9a0 \
  --strict
