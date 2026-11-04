from pathlib import Path

import pytest

from app.adapters.storage.local import LocalDocumentStorage


def test_reads_back_what_it_saved(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path)

    storage.save("tenant/abc.pdf", b"%PDF-1.7")

    assert storage.read("tenant/abc.pdf") == b"%PDF-1.7"


def test_reads_none_for_a_missing_key(tmp_path: Path) -> None:
    assert LocalDocumentStorage(tmp_path).read("tenant/none.pdf") is None


def test_keeps_the_first_file_under_a_key(tmp_path: Path) -> None:
    storage = LocalDocumentStorage(tmp_path)

    storage.save("tenant/abc.pdf", b"first")
    storage.save("tenant/abc.pdf", b"second")

    assert storage.read("tenant/abc.pdf") == b"first"


def test_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    LocalDocumentStorage(tmp_path).save("tenant/abc.pdf", b"content")

    assert [p.name for p in (tmp_path / "tenant").iterdir()] == ["abc.pdf"]


@pytest.mark.parametrize("key", ["../outside.pdf", "tenant/../../outside.pdf"])
def test_refuses_a_key_that_leaves_the_root(tmp_path: Path, key: str) -> None:
    storage = LocalDocumentStorage(tmp_path / "root")

    with pytest.raises(ValueError, match="leaves the storage root"):
        storage.save(key, b"content")
    with pytest.raises(ValueError, match="leaves the storage root"):
        storage.read(key)
