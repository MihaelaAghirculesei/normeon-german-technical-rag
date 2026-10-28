class NormeonError(Exception):
    """Base class for all typed application errors."""

    code: str = "internal_error"
    status_code: int = 500
    #: Sent as `Retry-After` when set: when the client may try again.
    retry_after_s: int | None = None


class DocumentNotFoundError(NormeonError):
    code = "document_not_found"
    status_code = 404


class NotAuthenticatedError(NormeonError):
    """No bearer token, or one that is malformed, forged or expired. One
    code for all of them: telling an attacker *why* a token was refused
    helps nobody else."""

    code = "not_authenticated"
    status_code = 401


class InvalidCredentialsError(NormeonError):
    """Login with an unknown email or a wrong password -- deliberately not
    distinguished, so the endpoint cannot be used to probe for accounts."""

    code = "invalid_credentials"
    status_code = 401


class ForbiddenError(NormeonError):
    """Authenticated, but the role does not allow this action."""

    code = "forbidden"
    status_code = 403


class UnsupportedFileTypeError(NormeonError):
    """An upload whose extension no parser handles (only .pdf and .docx)."""

    code = "unsupported_file_type"
    status_code = 415


class TooManyPagesError(NormeonError):
    """A readable document with more pages than ingestion accepts."""

    code = "too_many_pages"
    status_code = 413


class EncryptedDocumentError(NormeonError):
    """A password-protected PDF or Word file: nothing in it can be read."""

    code = "encrypted_document"
    status_code = 422


class MalformedDocumentError(NormeonError):
    """The file claims to be a PDF or Word file but cannot be opened as one."""

    code = "malformed_document"
    status_code = 422


class SuspiciousArchiveError(NormeonError):
    """A Word file (a zip archive) that would unpack to far more than it
    weighs -- the shape of a zip bomb."""

    code = "suspicious_archive"
    status_code = 422


class ChunkNotFoundError(NormeonError):
    """No chunk with that id for this tenant -- a chunk of another tenant
    is reported the same way, so the endpoint never confirms it exists."""

    code = "chunk_not_found"
    status_code = 404


class LlmTimeoutError(NormeonError):
    """The configured LLM provider did not answer within the configured
    timeout, even after retrying transient failures."""

    code = "llm_timeout"
    status_code = 504


class LlmUnavailableError(NormeonError):
    """The configured LLM provider could not be reached, or returned an
    error response (after retrying transient ones)."""

    code = "llm_unavailable"
    status_code = 502


class LlmMisconfiguredError(NormeonError):
    """`llm_provider` is set to something that needs settings which are
    not configured (e.g. `openai_compatible` without a base URL)."""

    code = "llm_misconfigured"
    status_code = 500


class EmbeddingMisconfiguredError(NormeonError):
    """`embedding_provider` is set to something that needs settings which
    are not configured (e.g. `e5_api` without a base URL). Same shape as
    `LlmMisconfiguredError` -- both are `get_*` dependency wiring errors,
    not request-time failures."""

    code = "embedding_misconfigured"
    status_code = 500


class DailyLimitReachedError(NormeonError):
    """The demo's daily question cap is spent (services/budget.py)."""

    code = "daily_limit_reached"
    status_code = 429

    def __init__(self, message: str, *, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
