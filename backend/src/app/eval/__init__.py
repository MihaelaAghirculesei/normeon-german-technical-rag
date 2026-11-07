"""The evaluation harness.

The code lives under ``app.eval`` (not a top-level ``eval/`` package) so
it stays inside the same ``ruff``/``mypy strict``/``pytest`` gate as the
rest of the app and can ``import`` from ``app.*`` cleanly. The *data* --
the question set, the generated JSON schema, and the run reports -- lives
under ``backend/eval/``; ``scripts/run_eval.py`` is the CLI entry point,
following the same "logic in src, entry points in scripts" split the rest
of the repo uses.
"""
