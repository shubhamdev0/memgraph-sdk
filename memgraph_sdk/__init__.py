"""Memgraph SDK - The official Python SDK for Memgraph, the memory graph for AI agents."""

__version__ = "0.2.0"

from .client import MemgraphClient

try:
    from .async_client import AsyncMemgraphClient
except ImportError:
    AsyncMemgraphClient = None  # httpx not installed

__all__ = ["MemgraphClient", "AsyncMemgraphClient", "__version__"]
