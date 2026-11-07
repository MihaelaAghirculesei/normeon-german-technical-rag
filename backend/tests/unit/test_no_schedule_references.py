"""The repository describes what the code does, not when it was written.

A label like "(Day 13)" or "Week 4" points at a schedule that no reader of
the repository has, so it explains nothing and goes stale. Every tracked
text file is checked; name the feature, the run or the decision instead.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATTERN = re.compile(r"\b(?:day|days|giorn[oi]|week) ?\d|fatto quando", re.IGNORECASE)
SKIP = {"frontend/package-lock.json", "backend/tests/unit/test_no_schedule_references.py"}


def _tracked_files() -> list[str]:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("needs a git checkout")
    out = subprocess.run(
        [git, "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [name for name in out.splitlines() if name not in SKIP]


def test_no_file_refers_to_a_schedule() -> None:
    hits = []
    for name in _tracked_files():
        try:
            text = (ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue  # binary, or deleted in the working tree
        for number, line in enumerate(text.splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{name}:{number}: {line.strip()}")

    assert not hits, "schedule references:\n" + "\n".join(hits)
