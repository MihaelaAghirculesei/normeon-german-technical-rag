from typing import Protocol


class DocumentStorage(Protocol):
    """Keeps the bytes of uploaded files, so a viewer can show the original."""

    def save(self, key: str, content: bytes) -> None:
        """Stores `content` under `key`. Keys are derived from the content
        hash, so a key that already exists holds the same bytes."""
        ...

    def read(self, key: str) -> bytes | None:
        """The bytes under `key`, or None if nothing is stored there."""
        ...
