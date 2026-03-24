## Summary

Briefly describe what this PR changes and why.

## Scope

- [ ] docs only
- [ ] release / proof tooling
- [ ] repo governance / policy
- [ ] tests only
- [ ] runtime code under `src/`
- [ ] other (describe below)

## Branch intent

- Source branch: `<branch-name>`
- Target branch: `<target-branch>`

- [ ] This PR is safe for `main`
- [ ] This PR is post-jury work and must stay outside `main`

Reference:
- [ ] I checked `docs/BRANCH_POLICY.md`

## Jury safety

- [ ] Does **not** modify the immutable jury tag
- [ ] Does **not** change the validated jury-safe baseline unless explicitly described below
- [ ] Does **not** merge `dev/post-jury` into `main`

If this PR touches runtime/QC behavior, explain exactly why it is still safe:

<!-- explain here -->

## Runtime invariants

Complete only if this PR touches runtime code, QC, exports, or validation logic.

- [ ] `09487_ok` remains `PASS / PASS / PASS`
- [ ] `11899` remains `WARN / WARN / WARN`
- [ ] limb swap remains warning-only and does not become `FAIL` by itself
- [ ] not applicable

If any invariant changes, explain explicitly:

<!-- explain here -->

## Runtime impact

- [ ] No files under `src/` are touched
- [ ] Files under `src/` are touched (list below)

Touched runtime files:
- `...`

QC / runtime impact summary:

<!-- explain here -->

## Proof / release bundle

Complete if this PR affects release proofs, jury assets, or proof tooling.

- [ ] not applicable
- [ ] proof bundle builder touched
- [ ] release/proof docs touched
- [ ] proof bundle regenerated locally

Bundle command used:

```bash
python scripts/build_proofs_bundle.py --tag <tag> --expected-tag-commit <commit> --strict
