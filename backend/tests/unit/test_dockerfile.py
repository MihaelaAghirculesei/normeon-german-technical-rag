"""Guards for both images -- backend/Dockerfile (local development) and
deploy/Dockerfile (the deployed demo): the files the app reads at runtime
are in them, and secrets stay out of the build context.

The app reads its prompts and price list relative to the backend root
(`Path(__file__).parents[3] / ...`). Until this test existed the image
copied only `src/`, so every chat request in a container failed on a
missing prompt file -- while every test passed, because tests run from the
source tree.
"""

import re
from pathlib import Path

import pytest
import yaml

BACKEND = Path(__file__).parents[2]
ROOT = BACKEND.parent
DOCKERFILES = {"backend": BACKEND / "Dockerfile", "deploy": ROOT / "deploy" / "Dockerfile"}
SRC = BACKEND / "src" / "app"
# Read only by evaluation scripts, which run from a checkout, not the image.
NOT_IN_IMAGE = {"eval"}


def _text(image: str) -> str:
    return DOCKERFILES[image].read_text(encoding="utf-8")


def _copied(image: str) -> set[str]:
    names: set[str] = set()
    for line in _text(image).splitlines():
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


images = pytest.mark.parametrize("image", sorted(DOCKERFILES))


@images
def test_every_file_the_app_reads_from_the_backend_root_is_in_the_image(image: str) -> None:
    needed = _read_at_runtime()
    copied = _copied(image)

    assert {"prompts", "pricing.yaml"} <= needed  # the scan itself works
    assert needed <= copied, f"not copied into the image: {needed - copied}"


@images
def test_the_app_is_installed_where_its_runtime_files_are(image: str) -> None:
    # `pip install .` would put a second copy in site-packages, whose
    # `parents[3]` is the Python lib directory, not the backend root.
    dockerfile = _text(image)

    assert re.search(r'pip install (--\S+ )*-e "?\.(\[\w+\])?"?$', dockerfile, re.MULTILINE)
    assert not re.search(r'pip install (--\S+ )*"?\.(\[\w+\])?"?$', dockerfile, re.MULTILINE)


@images
def test_migrations_ship_with_the_image(image: str) -> None:
    assert {"migrations", "alembic.ini"} <= _copied(image)


def test_secrets_and_the_virtualenv_stay_out_of_the_build_context() -> None:
    ignored = (BACKEND / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".venv/" in ignored


@images
def test_the_image_does_not_run_as_root(image: str) -> None:
    lines = _text(image).splitlines()
    users = [line.split()[1] for line in lines if line.startswith("USER ")]

    assert users and users[-1] not in {"root", "0"}


def test_volumes_in_the_users_home_are_created_by_the_user() -> None:
    # A fresh named volume takes the owner of the image directory it is
    # mounted on, and root's if that directory does not exist.
    compose = yaml.safe_load((BACKEND.parent / "docker-compose.yml").read_text(encoding="utf-8"))
    home = "/home/normeon/"
    targets = [
        volume.split(":")[1]
        for volume in compose["services"]["backend"]["volumes"]
        if volume.split(":")[1].startswith(home)
    ]
    dockerfile = (BACKEND / "Dockerfile").read_text(encoding="utf-8")
    after_user = dockerfile[dockerfile.rindex("\nUSER ") :]

    assert targets
    for target in targets:
        assert f"mkdir -p {target}" in after_user, target


def test_the_development_image_runs_the_models_in_process() -> None:
    assert 'pip install -e ".[local]"' in _text("backend")


def test_the_deployed_image_installs_no_torch() -> None:
    dockerfile = _text("deploy")
    runs = [line for line in dockerfile.splitlines() if line.lstrip().startswith(("RUN", "&&"))]

    assert not any("torch" in line or "[local]" in line for line in runs)
    assert "EMBEDDING_PROVIDER=gemini" in dockerfile
    assert "RERANKER_PROVIDER=noop" in dockerfile


def test_the_deployed_image_serves_the_built_frontend() -> None:
    dockerfile = _text("deploy")

    assert "COPY --from=frontend /frontend/dist/frontend/browser ./static" in dockerfile
    assert "STATIC_DIR=/app/static" in dockerfile


def test_the_deploy_build_context_lets_in_only_what_the_image_needs() -> None:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    allowed = {line[1:] for line in lines if line.startswith("!")}

    assert "**" in lines  # everything not let in below is excluded
    assert {"deploy/start.sh", "frontend/", "backend/src/", "backend/migrations/"} <= allowed
    assert {"**/.env", "**/.env.*", "frontend/node_modules/"} <= set(lines)
    assert not any(path.startswith(("corpus", "backend/eval", "backend/.venv")) for path in allowed)


def test_the_start_script_migrates_and_seeds_before_serving() -> None:
    raw = (ROOT / "deploy" / "start.sh").read_bytes()
    script = raw.decode("utf-8")
    commands = [line.split()[0] for line in script.splitlines() if line and line[0] not in "# "]

    assert b"\r" not in raw  # sh in the image would choke on CRLF
    assert commands[commands.index("alembic") :] == ["alembic", "python", "exec"]
    assert '--port "${PORT:-8000}"' in script


def test_the_deployed_image_turns_the_rate_limits_on() -> None:
    dockerfile = _text("deploy")
    limits = dict(re.findall(r"(RATE_LIMIT_\w+)=(\d+)", dockerfile))

    assert set(limits) == {
        "RATE_LIMIT_LOGIN_PER_MINUTE",
        "RATE_LIMIT_QUESTIONS_PER_MINUTE",
        "RATE_LIMIT_UPLOADS_PER_HOUR",
    }
    assert all(int(value) > 0 for value in limits.values())


def test_the_deployed_image_caps_the_questions_a_day() -> None:
    match = re.search(r"DAILY_QUESTION_LIMIT=(\d+)", _text("deploy"))

    assert match and int(match.group(1)) > 0
