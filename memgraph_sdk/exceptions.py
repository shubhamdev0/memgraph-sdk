"""Custom exception hierarchy for the Memgraph SDK.

Allows callers to catch specific error types and implement proper error recovery.
"""

from typing import Optional


class MemgraphError(Exception):
    """Base exception for all Memgraph SDK errors."""

    def __init__(self, message: str = "", status_code: Optional[int] = None, response_body: Optional[dict] = None):
        self.status_code = status_code
        self.response_body = response_body or {}
        super().__init__(message)


class MemgraphAuthError(MemgraphError):
    """Raised when authentication fails (401/403)."""
    pass


class MemgraphConnectionError(MemgraphError):
    """Raised when the server is unreachable or times out."""
    pass


class MemgraphRateLimitError(MemgraphError):
    """Raised when rate limit is exceeded (429). Check retry_after_seconds."""

    def __init__(self, message: str = "", retry_after: int = 60, **kwargs):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)


class MemgraphAPIError(MemgraphError):
    """Raised for server errors (5xx) or unexpected responses."""
    pass


class MemgraphValidationError(MemgraphError):
    """Raised for client-side validation errors (4xx, excluding auth/rate limit)."""
    pass
