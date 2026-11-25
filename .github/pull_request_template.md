## What

<!-- One or two sentences: what does this PR change. -->

## Why

<!-- The reason this change is needed. Link an issue if one exists. -->

## Checklist

- [ ] `ruff check .`, `mypy src` and `pytest` pass locally (from `backend/`)
- [ ] Eval numbers in docs come from a real run, not estimates
- [ ] `docs/EVALUATION.md` / ADR updated if this changes retrieval, scoring or an architectural decision
- [ ] New or changed workflow: every secret it references is set (`gh secret list`) before it is merged or enabled, and it fails with a clear message when one is empty
