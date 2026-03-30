"""Memgraph SDK - The official Python SDK for Memgraph, the memory graph for AI agents."""

__version__ = "0.6.0"

from .client import MemgraphClient
from .exceptions import (
    MemgraphAPIError,
    MemgraphAuthError,
    MemgraphConnectionError,
    MemgraphError,
    MemgraphRateLimitError,
    MemgraphValidationError,
)

try:
    from .async_client import AsyncMemgraphClient
except ImportError:
    AsyncMemgraphClient = None  # httpx not installed

try:
    from .middleware import CognitiveSidecar
except ImportError:
    CognitiveSidecar = None

try:
    from .agent_memory import MemgraphMemory
except ImportError:
    MemgraphMemory = None

try:
    from .openai_agents import (
        MemgraphAgentHooks,
        MemgraphRunHooks,
        create_memgraph_agent,
        memgraph_instructions,
        memgraph_tools,
    )
except ImportError:
    MemgraphAgentHooks = None
    MemgraphRunHooks = None
    create_memgraph_agent = None
    memgraph_instructions = None
    memgraph_tools = None

__all__ = [
    "MemgraphClient",
    "AsyncMemgraphClient",
    "CognitiveSidecar",
    "MemgraphMemory",
    "MemgraphError",
    "MemgraphAPIError",
    "MemgraphAuthError",
    "MemgraphConnectionError",
    "MemgraphRateLimitError",
    "MemgraphValidationError",
    "MemgraphAgentHooks",
    "MemgraphRunHooks",
    "create_memgraph_agent",
    "memgraph_instructions",
    "memgraph_tools",
    "__version__",
]
