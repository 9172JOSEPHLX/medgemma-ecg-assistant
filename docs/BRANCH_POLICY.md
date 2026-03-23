## 2) `docs/BRANCH_POLICY.md`
```md
# Branch Policy

## Purpose

This document defines the repository branch safety rules for Appli_MedGemma_ECG.

The goal is to keep the jury-validated baseline stable while allowing post-jury work to continue safely.

## Immutable source-of-truth

The immutable jury reference is:

- Tag: `v0.1.1-kaggle-jury-20260312`
- Commit: `49de9a0`

This tag must never be modified, moved, or replaced.

## Branch roles

### `main`
Purpose:
- jury-safe branch,
- releasable branch,
- documentation and maintenance via PR only.

Allowed changes:
- docs,
- release notes,
- jury appendix,
- `.gitignore`,
- maintenance/scripts that do not alter validated runtime behavior.

Forbidden during jury-safe phase:
- direct pushes,
- unreviewed runtime changes,
- post-jury feature merges,
- `dev/post-jury -> main`.

### `dev/maintenance-jury-safe`
Purpose:
- safe maintenance branch for changes that are acceptable for `main`.

Typical examples:
- docs updates,
- PR templates,
- release proof scripts,
- repo hygiene.

This branch is the preferred source for safe PRs into `main`.

### `dev/post-jury`
Purpose:
- post-jury feature development,
- digitization MVP/productization,
- non-critical-path technical consolidation.

Typical examples:
- digitization adapters,
- CLI feature work,
- future OCR/TorchScript wiring,
- broader post-jury R&D.

This branch must not be merged into `main` during jury-safe closure.

## Pull request rules

### Allowed PRs to `main`
Allowed:
- `dev/maintenance-jury-safe -> main`
- dedicated docs-only branches -> `main`

Not allowed during jury-safe closure:
- `dev/post-jury -> main`

### PR requirements
For changes targeting `main`:
- PR required,
- atomic scope,
- no unrelated file changes,
- clear title and rationale,
- explicit statement whether runtime code is touched,
- explicit reference to the immutable tag when relevant.

## Review rules

Any PR to `main` should state:
- scope,
- risk,
- whether `src/` is touched,
- whether QC/runtime behavior is affected,
- validation performed.

If runtime code is touched, the PR is not considered maintenance-only.

## Runtime protection rules

Non-negotiable:
- do not break QC stable contract,
- do not change baseline semantics:
  - `09487_ok` stays `PASS / PASS / PASS`
  - `11899` stays `WARN / WARN / WARN`
  - SWAP warning remains acquisition-only, never FAIL by itself
- do not modify the immutable tag.

## Recommended GitHub protection for `main`

Recommended settings:
- require pull request before merging,
- require conversation resolution,
- require linear history,
- disable force-push,
- disable branch deletion,
- require checks if CI is configured.

## Practical workflow

### Safe maintenance/doc flow
1. branch from `main`
2. make small isolated change
3. open PR to `main`
4. merge after review/checks

### Post-jury feature flow
1. work on `dev/post-jury`
2. keep changes atomic
3. validate locally
4. do not target `main` until jury-safe phase is explicitly over

## Summary

- `main` = stable, reviewable, releasable
- `dev/maintenance-jury-safe` = safe path to `main`
- `dev/post-jury` = feature branch, isolated from `main`
- immutable tag = final jury code truth