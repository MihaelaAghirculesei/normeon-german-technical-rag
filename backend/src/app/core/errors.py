class NormeonError(Exception):
    """Base class for all typed application errors."""

    code: str = "internal_error"
    status_code: int = 500


class DocumentNotFoundError(NormeonError):
    code = "document_not_found"
    status_code = 404


class UnsupportedFileTypeError(NormeonError):
    """An upload whose extension no parser handles (only .pdf and .docx)."""

    code = "unsupported_file_type"
    status_code = 415


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
