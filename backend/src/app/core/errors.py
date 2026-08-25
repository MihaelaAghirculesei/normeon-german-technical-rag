class NormeonError(Exception):
    """Base class for all typed application errors."""

    code: str = "internal_error"
    status_code: int = 500


class DocumentNotFoundError(NormeonError):
    code = "document_not_found"
    status_code = 404
