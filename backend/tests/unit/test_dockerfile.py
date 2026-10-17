"""Guards for the backend image: the files the app reads at runtime are in
it, and secrets stay out of the build context.

The app reads its prompts and price list relative to the backend root
(`Path(__file__).parents[3] / ...`). Until this test existed the image
copied only `src/`, so every chat request in a container failed on a
missing prompt file -- while every test passed, because tests run from the
source tree.
"""

import re
from pathlib import Path

BACKEND = Path(__file__).parents[2]
SRC = BACKEND / "src" / "app"
# Read only by evaluation scripts, which run from a checkout, not the image.
NOT_IN_IMAGE = {"eval"}


def _copied() -> set[str]:
    names: set[str] = set()
    for line in (BACKEND / "Dockerfile").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts[:1] == ["COPY"]:
            names.update(Path(p).name for p in parts[1:-1])
    return names


def _read_at_runtime() -> set[str]:
    pattern = re.compile(r'parents\[3\]\s*/\s*"([^"]+)"')
    found: set[str] = set()
    for module in SRC.rglob("*.py"):
        found.update(pattern.findall(module.read_text(encoding="utf-8")))
    return found - NOT_IN_IMAGE


def test_every_file_the_app_reads_from_the_backend_root_is_in_the_image() -> None:
    needed = _read_at_runtime()

    assert {"prompts", "pricing.yaml"} <= needed  # the scan itself works
    assert needed <= _copied(), f"not copied into the image: {needed - _copied()}"


def test_the_app_is_installed_where_its_runtime_files_are() -> None:
    # `pip install .` would put a second copy in site-packages, whose
    # `parents[3]` is the Python lib directory, not the backend root.
    dockerfile = (BACKEND / "Dockerfile").read_text(encoding="utf-8")

    assert re.search(r"pip install (--\S+ )*-e \.$", dockerfile, re.MULTILINE)
    assert not re.search(r"pip install (--\S+ )*\.$", dockerfile, re.MULTILINE)


def test_migrations_ship_with_the_image() -> None:
    assert {"migrations", "alembic.ini"} <= _copied()


def test_secrets_and_the_virtualenv_stay_out_of_the_build_context() -> None:
    ignored = (BACKEND / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".venv/" in ignored
