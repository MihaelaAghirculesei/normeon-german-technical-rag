"""The deployed image installs without the `local` extra (ADR 0005): the
app must start, and serve with Gemini embeddings and no reranker, when
torch and sentence-transformers cannot be imported at all."""

import subprocess
import sys
import textwrap

_PROGRAM = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in {"torch", "sentence_transformers", "transformers"}:
                raise ImportError(f"blocked: {name}")
            return None

    sys.meta_path.insert(0, _Block())

    import app.main  # noqa: F401
    from app.api import deps

    deps.settings.embedding_provider = "gemini"
    deps.settings.embedding_api_key = "key"
    deps.settings.embedding_model = "gemini-embedding-2"
    deps.settings.reranker_provider = "noop"
    print(deps.get_embedder().name, deps.get_reranker().name)
    """
)


def test_the_app_starts_without_torch() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _PROGRAM], capture_output=True, text=True, timeout=120
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        "gemini:gemini-embedding-2",
        "noop",
    ]
