"""Memgraph SDK - The official Python SDK for Memgraph, the memory graph for AI agents."""

__version__ = "0.4.0"

from .client import MemgraphClient
from .exceptions import (
    MemgraphError,
    MemgraphAPIError,
    MemgraphAuthError,
    MemgraphConnectionError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)

try:
    from .async_client import AsyncMemgraphClient
except ImportError:
    AsyncMemgraphClient = None  # httpx not installed

__all__ = [
    "MemgraphClient",
    "AsyncMemgraphClient",
    "MemgraphError",
    "MemgraphAPIError",
    "MemgraphAuthError",
    "MemgraphConnectionError",
    "MemgraphRateLimitError",
    "MemgraphValidationError",
    "__version__",
]
