# Memgraph SDK

**Memory that thinks.** The official Python SDK for **[Memgraph](https://memgraph.ai)** — the cognitive memory layer for AI agents.

[![PyPI version](https://badge.fury.io/py/memgraph-sdk.svg)](https://pypi.org/project/memgraph-sdk/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Not a vector store with a wrapper. A three-layer cognitive engine that distills raw events into episodes and crystallizes them into beliefs — with background Cognitive Dreaming that improves memory while your agents sleep.

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

## Quick Start

```python
from memgraph_sdk import MemgraphClient

client = MemgraphClient(api_key="mg_your_api_key")

# Store a memory
client.add("User prefers dark mode and uses PyTorch.", user_id="alice")

# Store a belief directly
client.remember("Prefers PyTorch over TensorFlow", user_id="alice", category="preference")

# Search for relevant context
result = client.search("What does this user prefer?", user_id="alice")
print(result)
```

That's it. Two lines to set up, then store and recall. The `tenant_id` is resolved automatically from your API key.

## Async Client

```python
from memgraph_sdk import AsyncMemgraphClient

async with AsyncMemgraphClient(api_key="mg_your_api_key") as client:
    await client.add("User prefers dark mode", user_id="alice")
    result = await client.search("user preferences", user_id="alice")
```

## MCP Server (Claude / Cursor)

One command to give your AI IDE persistent memory:

```bash
memgraph setup --key mg_your_api_key
```

This auto-detects your IDE and writes the MCP config. Or configure manually:

```json
{
  "mcpServers": {
    "memgraph": {
      "command": "python3",
      "args": ["-m", "memgraph_sdk.mcp"],
      "env": {
        "MEMGRAPH_API_KEY": "mg_your_api_key"
      }
    }
  }
}
```

## Memory Intelligence API

```python
# Memory health metrics
health = client.health(user_id="alice")

# Detect contradictions
contradictions = client.contradictions(user_id="alice")

# Evaluate retrieval quality
evaluation = client.evaluate(query="test query", user_id="alice")

# Cognitive Integrity Score (0-100)
score = client.mcis(user_id="alice")

# Run benchmarks
result = client.benchmark(scenario="contradiction_storm")
```

## CLI

```bash
# Set up MCP for your IDE (auto-detects Cursor, Claude, VS Code)
memgraph setup --key mg_your_api_key

# Store a memory
memgraph remember "We decided to use PostgreSQL" -c decision

# Search memories
memgraph recall "database choice"

# Check connection
memgraph status
```

## Configuration

### Cloud (default)

```python
# Connects to https://api.memgraph.ai/v1 by default
client = MemgraphClient(api_key="mg_your_key")
```

### Self-hosted

```python
client = MemgraphClient(
    api_key="mg_your_key",
    base_url="http://your-server:8001/v1",
)
```

Or via environment variable:

```bash
export MEMGRAPH_API_URL=http://your-server:8001/v1
export MEMGRAPH_API_KEY=mg_your_key
```

### URL Resolution Priority

1. `base_url` parameter (highest priority)
2. `MEMGRAPH_API_URL` environment variable
3. `https://api.memgraph.ai/v1` (cloud default)

## How It Works

```
Raw Input → Events → Episodes → Beliefs
              │          │          │
          (short-term) (grouped)  (long-term)
```

- **Events**: Raw, immutable records with vector embeddings
- **Episodes**: Auto-grouped sequences with LLM summaries
- **Beliefs**: Extracted facts, preferences, decisions with confidence scores
- **Cognitive Dreaming**: Background worker that consolidates, deduplicates, and resolves contradictions while your agents sleep

## Integrations

Works with any AI framework:

- **LangChain** — Memory, Retriever, Toolkit components
- **OpenAI** — Function-calling agent and Assistants API
- **CrewAI** — Search and Remember tools
- **LlamaIndex** — Retriever and ToolSpec

See the [examples/](examples/) directory for complete integration examples.

## Documentation

- [Quick Start](https://memgraph.ai/docs)
- [SDK Reference](https://memgraph.ai/docs)
- [Self-Hosting Guide](https://memgraph.ai/docs)
- [API Reference](https://memgraph.ai/docs)

## License

MIT License. See [LICENSE](LICENSE) for details.
