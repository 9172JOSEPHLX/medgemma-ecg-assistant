## Summary
Describe the change in 2-4 lines.

## Scope
- [ ] docs only
- [ ] maintenance only
- [ ] runtime code touched
- [ ] tests added/updated

## Branch intent
Source branch:
Target branch:

## Jury safety
- [ ] This PR does not modify the immutable tag `v0.1.1-kaggle-jury-20260312`
- [ ] This PR does not merge `dev/post-jury` into `main`
- [ ] This PR is safe for `main`
- [ ] This PR changes only docs/maintenance
- [ ] This PR touches runtime behavior (explain below if checked)

## Runtime impact
Files under `src/` touched:
- none / list them here

QC invariant impact:
- [ ] no impact
- [ ] impact reviewed and explained

If runtime is touched, explain:
- what changed,
- why it is safe,
- how baseline semantics are preserved.

## Validation
Commands run:
```bash
# example
pytest -q