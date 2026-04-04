<p align="center">
  <img src="assets/logo.png" alt="Memgraph AI" width="400">
</p>

<h3 align="center">Memory that helps AI agents learn from their mistakes</h3>

<p align="center">
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/v/memgraph-sdk?color=%2334D058&label=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/memgraph-sdk/"><img src="https://img.shields.io/pypi/dm/memgraph-sdk" alt="Downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python"></a>
  <a href="https://github.com/shubhamdev0/memgraph-sdk/actions"><img src="https://github.com/shubhamdev0/memgraph-sdk/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://memgraph.ai">Website</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/tree/main/examples">Examples</a> ·
  <a href="https://github.com/shubhamdev0/memgraph-sdk/issues">Issues</a>
</p>

---

Not a vector store with a wrapper. A three-layer cognitive engine that distills raw events into episodes, crystallizes them into beliefs, and tracks how those beliefs evolve — with background consolidation that improves memory while your agents sleep.

## Table of Contents

- [Installation](#installation)
- [Give your agent a memory in 30 seconds](#give-your-agent-a-memory-in-30-seconds)
- [Authentication](#authentication)
- [Core Methods](#core-methods)
- [Error Handling](#error-handling)
- [Async Client](#async-client)
- [MCP Server (Claude / Cursor)](#mcp-server-claude--cursor)
- [CLI](#cli)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [Integrations](#integrations)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

```bash
pip install memgraph-sdk
```

With optional extras:

```bash
pip install "memgraph-sdk[async]"   # Async client (httpx)
pip install "memgraph-sdk[mcp]"     # MCP server for Claude/Cursor
pip install "memgraph-sdk[all]"     # Everything
```

## Give your agent a memory in 30 seconds

```python
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key="mg_your_api_key")

# Store a memory (immediately searchable)
mg.remember("Customer prefers dark mode and uses PyTorch", user_id="alice")

# Search memories
result = mg.search("What does Alice prefer?", user_id="alice")
print(result["results"][0]["content"])
# → "Customer prefers dark mode and uses PyTorch" (score: 0.78)

# Get all beliefs for a user
beliefs = mg.get_beliefs(user_id="alice")
```

Three lines to set up. Store and search immediately. The `tenant_id` is resolved automatically from your API key.

## Authentication

Set your API key as an environment variable so you never have to worry about committing credentials:

```bash
export MEMGRAPH_API_KEY=mg_your_api_key
```

```python
import os
from memgraph_sdk import MemgraphClient

mg = MemgraphClient(api_key=os.environ["MEMGRAPH_API_KEY"])
```

Or pass it directly (not recommended for production):

```python
mg = MemgraphClient(api_key="mg_your_api_key")
```

Get your API key at [memgraph.ai](https://memgraph.ai).

## Core Methods

```python
# Store (immediately searchable with vector embedding)
mg.remember("User prefers dark mode", user_id="alice", category="preference")

# Store (async pipeline — extraction + consolidation)
mg.add("Full conversation text here", user_id="alice")

# Search (returns scored results with semantic similarity)
result = mg.search("UI preferences", user_id="alice")
# → {"results": [{"content": "...", "score": 0.76, "metadata": {...}}], "total": 1}

# Get all beliefs for a user
beliefs = mg.get_beliefs(user_id="alice", limit=50)

# Health check
status = mg.ping()
```

### Context manager

```python
with MemgraphClient(api_key="mg_your_key") as mg:
    mg.remember("User likes Python", user_id="alice")
    # Session is automatically closed when block exits
```

## Error Handling

```python
from memgraph_sdk import MemgraphClient
from memgraph_sdk.exceptions import (
    MemgraphAuthError,
    MemgraphConnectionError,
    MemgraphRateLimitError,
    MemgraphValidationError,
    MemgraphAPIError,
)

mg = MemgraphClient(api_key="mg_your_key")

try:
    result = mg.search("query", user_id="alice")
except MemgraphAuthError:
    # Invalid API key (401/403)
    print("Check your MEMGRAPH_API_KEY")
except MemgraphRateLimitError as e:
    # Too many requests (429) — retry after e.retry_after seconds
    print(f"Rate limited. Retry in {e.retry_after}s")
except MemgraphConnectionError:
    # Server unreachable or timeout
    print("Cannot reach Memgraph server")
except MemgraphValidationError as e:
    # Bad request (422) — check your parameters
    print(f"Validation error: {e}")
except MemgraphAPIError as e:
    # Server error (5xx) — transient, retried automatically
    print(f"Server error {e.status_code}: {e}")
```

The SDK automatically retries transient errors (500, 502, 503, 504) with exponential backoff. Auth and validation errors are raised immediately.

## Async Client

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(api_key="mg_your_api_key") as mg:
    mg.remember("User prefers dark mode", user_id="alice")
    result = await mg.search("preferences", user_id="alice")
```

Requires: `pip install "memgraph-sdk[async]"`

## MCP Server (Claude / Cursor)

Give your AI IDE persistent memory with one command:

```bash
memgraph setup --key mg_your_api_key
```

Auto-detects Cursor, Claude Desktop, VS Code. Or configure manually:

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "python3",
      "args": ["-m", "memgraph_sdk.mcp"],
      "env": { "MEMGRAPH_API_KEY": "mg_your_api_key" }
    }
  }
}
```

## CLI

```bash
memgraph setup --key mg_your_api_key    # Set up MCP for your IDE
memgraph remember "We chose PostgreSQL"  # Store a memory
memgraph recall "database choice"        # Search memories
memgraph status                          # Check connection
```

## Configuration

### Cloud (default)

```python
mg = MemgraphClient(api_key="mg_your_key")
# Connects to https://api.memgraph.ai/v1
```

### Self-hosted

```python
mg = MemgraphClient(
    api_key="mg_your_key",
    base_url="http://your-server:8001/v1",
)
```

### Environment variables

```bash
export MEMGRAPH_API_KEY=mg_your_key
export MEMGRAPH_API_URL=http://your-server:8001/v1  # optional
```

**URL resolution priority:**
1. `base_url` parameter (highest)
2. `MEMGRAPH_API_URL` environment variable
3. `https://api.memgraph.ai/v1` (default)

## How It Works

```
Raw Input → Events → Episodes → Beliefs
              │          │          │
          (short-term) (grouped)  (long-term)
                                     │
                              Cognitive Dreaming
                         (consolidation while idle)
```

- **Events** — Raw, immutable records with vector embeddings
- **Episodes** — Auto-grouped sequences with LLM summaries
- **Beliefs** — Extracted facts, preferences, decisions with confidence scores and types (fact / belief / tenet)
- **Cognitive Dreaming** — Background worker that consolidates, deduplicates, and resolves contradictions

## Integrations

Works with any AI framework. See [examples/](examples/) for runnable code.

| Framework | What's included | Status |
|---|---|---|
| [OpenAI](examples/integrations/openai_integration.py) | Function calling agent with memory | ✅ Tested |
| [LangChain](examples/integrations/langchain_integration.py) | Memory, Retriever, Toolkit | ✅ Tested |
| [CrewAI](examples/integrations/crewai_integration.py) | Search and Remember tools | ✅ Tested |
| [LlamaIndex](examples/integrations/llamaindex_integration.py) | Retriever and ToolSpec | — |

## Examples

All examples tested against the production API (`api.memgraph.ai`):

| Example | Description |
|---|---|
| [quick_start.py](examples/quick_start.py) | Store, search, update — takes 2 minutes |
| [agent.py](examples/agent.py) | Interactive chat agent with OpenAI + memory |
| [sdk_demo.py](examples/sdk_demo.py) | Core SDK operations in 30 lines |

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Security

Report vulnerabilities to **security@memgraph.ai**. See [SECURITY.md](SECURITY.md).

## License

MIT — see [LICENSE](LICENSE).
