import os
import tempfile
from functools import lru_cache
from pathlib import Path

from app.core.config import settings


class LocalDocumentStorage:
    """Files on a local volume. On a host without a persistent disk they are
    lost at restart, and the citation panel falls back to the text view."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, key: str) -> Path:
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root.resolve()):
            raise ValueError(f"storage key leaves the storage root: {key!r}")
        return path

    def save(self, key: str, content: bytes) -> None:
        path = self._path(key)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written aside and moved, so a reader never sees half a file.
        handle, temporary = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(content)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def read(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None


@lru_cache
def default_storage() -> LocalDocumentStorage:
    return LocalDocumentStorage(settings.document_storage_dir)
