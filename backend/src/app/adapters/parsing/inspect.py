"""Check an uploaded file before anything is stored (Giorno 26).

Opening a file is cheap; parsing, chunking and embedding it is not. So the
upload endpoint asks here first whether the file can be read at all, and
refuses with a clear, typed error what would otherwise only end up as a
document stuck in "failed":
  - a PDF that does not open, has no pages, or has more pages than
    ingestion accepts;
  - a password-protected PDF or Word file;
  - a Word file that is not a valid .docx archive, or that would unpack to
    far more than it weighs (a zip bomb).

Only metadata is read: the PDF's page table, the zip's central directory.
Nothing is decompressed here, which is what makes the zip-bomb check safe.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import pymupdf

from app.core.errors import (
    EncryptedDocumentError,
    MalformedDocumentError,
    SuspiciousArchiveError,
    TooManyPagesError,
)

# Word encrypts a .docx by wrapping it in an OLE compound file, which is
# not a zip archive at all; this is its signature.
_OLE_SIGNATURE = bytes.fromhex("d0cf11e0a1b11ae1")
_DOCX_BODY = "word/document.xml"
_RATIO_CHECK_MIN_BYTES = 2**20


@dataclass(frozen=True, slots=True)
class UploadLimits:
    max_pages: int
    max_uncompressed_bytes: int
    max_compression_ratio: int


def inspect_upload(suffix: str, content: bytes, limits: UploadLimits) -> None:
    """Raise a typed error if `content` (a `suffix` file) cannot or must
    not be ingested; return quietly otherwise. `suffix` is lower-case and
    already known to be supported."""
    if suffix == ".pdf":
        _inspect_pdf(content, limits)
    elif suffix == ".docx":
        _inspect_docx(content, limits)


def _inspect_pdf(content: bytes, limits: UploadLimits) -> None:
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise MalformedDocumentError("the file is not a readable PDF") from exc
    with doc:
        if doc.needs_pass:
            raise EncryptedDocumentError(
                "the PDF is password-protected; upload an unprotected copy"
            )
        if doc.page_count == 0:
            raise MalformedDocumentError("the PDF has no pages")
        if doc.page_count > limits.max_pages:
            raise TooManyPagesError(
                f"the PDF has {doc.page_count} pages; at most {limits.max_pages} are accepted"
            )


def _inspect_docx(content: bytes, limits: UploadLimits) -> None:
    if content.startswith(_OLE_SIGNATURE):
        raise EncryptedDocumentError(
            "the Word file is password-protected; upload an unprotected copy"
        )
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise MalformedDocumentError("the file is not a readable Word document") from exc
    with archive:
        members = archive.infolist()
        if _DOCX_BODY not in {m.filename for m in members}:
            raise MalformedDocumentError("the archive has no Word document body")
        # zipfile never yields more than a member's declared size, so the
        # declared sizes are a real bound on what parsing will unpack.
        total = sum(m.file_size for m in members)
        if total > limits.max_uncompressed_bytes:
            raise SuspiciousArchiveError(
                f"the Word file would unpack to {total // 2**20} MB; "
                f"at most {limits.max_uncompressed_bytes // 2**20} MB are accepted"
            )
        for member in members:
            # Small parts of a real document (styles, content types) can be
            # very repetitive; a bomb, by definition, unpacks to a lot.
            if member.file_size <= _RATIO_CHECK_MIN_BYTES:
                continue
            ratio = member.file_size / max(member.compress_size, 1)
            if ratio > limits.max_compression_ratio:
                raise SuspiciousArchiveError(
                    f"{member.filename} is compressed {ratio:.0f}:1; "
                    "that is the shape of a zip bomb, not of a document"
                )
